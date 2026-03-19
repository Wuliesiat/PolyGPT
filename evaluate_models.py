#!/usr/bin/env python3
"""
涂层LLM评估系统
功能：
1. 让多个待评估模型回答问题（不提供参考文献）
2. 使用裁判模型根据金标准答案评分（0-10分）
3. 每个回答评分3次，计算平均分和误差
4. 生成评估报告和柱形图
"""

import json
import time
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class ModelConfig:
    """模型配置"""
    name: str  # 显示名称
    api_key: str
    base_url: str
    model_name: str
    
@dataclass
class EvaluationResult:
    """单个问题的评估结果"""
    question: str
    category: str
    gold_answer: str
    model_answer: str
    scores: List[float] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    source_literature: str = ""

class ModelEvaluator:
    def __init__(self, config_path: str = "eval_config.json"):
        """初始化评估器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 裁判模型配置
        judge_config = self.config["judge_model"]
        self.judge_client = OpenAI(
            api_key=judge_config["api_key"],
            base_url=judge_config["base_url"]
        )
        self.judge_model = judge_config["model_name"]
        
        # 待评估模型配置
        self.eval_models: List[ModelConfig] = []
        for m in self.config["eval_models"]:
            self.eval_models.append(ModelConfig(
                name=m["name"],
                api_key=m["api_key"],
                base_url=m["base_url"],
                model_name=m["model_name"]
            ))
        
        # 评估参数
        self.num_scoring_rounds = self.config.get("num_scoring_rounds", 3)
        self.request_delay = self.config.get("request_delay", 1)
        self.answer_temperature = self.config.get("answer_temperature", 0.7)
        self.judge_temperature = self.config.get("judge_temperature", 0.3)
        # 并行度：0 或 1 表示单线程，>1 表示线程池大小（建议 4–16，注意 API 限流）
        self.max_workers = self.config.get("max_workers", 6)
        self._print_lock = threading.Lock()
        
    def get_answer_prompt(self) -> str:
        """生成回答问题的提示词（无参考文献）"""
        return """You are an expert professor in coating materials science and engineering.

Please answer the following research question based on your knowledge and expertise.
Provide a comprehensive, detailed, and scientifically accurate answer.

Your answer should:
1. Be technically precise with specific details
2. Include relevant quantitative data when applicable
3. Cover the scientific principles and mechanisms involved
4. Be written in clear academic English
5. Be approximately 300-500 words

IMPORTANT: Answer based solely on your knowledge. Do not mention that you don't have access to specific literature or references.

Question: {question}

Please provide your answer:"""

    def get_judge_prompt(self) -> str:
        """生成裁判评分的提示词"""
        return """You are an expert evaluator assessing the quality of answers about coating materials science.

Your task is to compare a MODEL ANSWER against a GOLD STANDARD ANSWER and assign a score from 0-10.

## Scoring Criteria:

- **10**: Perfect answer, covers all key points with equal or better detail
- **8-9**: Excellent answer, covers most key points accurately with good detail
- **6-7**: Good answer, covers main concepts but missing some details or minor inaccuracies
- **4-5**: Acceptable answer, basic understanding but significant gaps or inaccuracies
- **2-3**: Poor answer, major errors or missing critical information
- **0-1**: Very poor or irrelevant answer

## Evaluation Dimensions:
1. **Accuracy**: Are the facts, data, and mechanisms correct?
2. **Completeness**: Does it cover the key aspects (synthesis, structure, properties, mechanism, application)?
3. **Technical Depth**: Are specific details (compositions, parameters, quantitative data) provided?
4. **Scientific Reasoning**: Is the explanation logical and well-connected?

## Question:
{question}

## Gold Standard Answer:
{gold_answer}

## Model Answer to Evaluate:
{model_answer}

## Your Response:
Provide your evaluation in the following JSON format:
{{
    "score": <0-10>,
    "reasoning": "<brief explanation of the score in 2-3 sentences>"
}}

Only output the JSON, nothing else."""

    def call_model(self, client: OpenAI, model: str, messages: List[Dict], 
                   temperature: float = 0.7, max_retries: int = 5) -> Optional[str]:
        """调用模型API（带指数退避重试）"""
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e)
                # 429 限流 或 5xx 服务端错误 → 重试
                is_retryable = ("429" in err_str or "500" in err_str or 
                                "502" in err_str or "503" in err_str or 
                                "负载" in err_str or "rate" in err_str.lower() or
                                "timeout" in err_str.lower())
                if is_retryable and attempt < max_retries:
                    wait = min(2 ** attempt * 2, 60)  # 2s, 4s, 8s, 16s, 32s, 最多60s
                    with self._print_lock:
                        print(f"    API调用错误 (第{attempt+1}次，{wait}s后重试): {err_str[:120]}")
                    time.sleep(wait)
                else:
                    with self._print_lock:
                        print(f"    API调用错误 (已放弃): {err_str[:200]}")
                    return None
        return None

    def get_model_answer(self, model_config: ModelConfig, question: str) -> Optional[str]:
        """让待评估模型回答问题"""
        client = OpenAI(
            api_key=model_config.api_key,
            base_url=model_config.base_url
        )
        
        prompt = self.get_answer_prompt().format(question=question)
        messages = [{"role": "user", "content": prompt}]
        
        return self.call_model(client, model_config.model_name, messages, 
                              self.answer_temperature)

    def judge_answer(self, question: str, gold_answer: str, 
                     model_answer: str) -> Optional[Tuple[float, str]]:
        """裁判模型评分"""
        prompt = self.get_judge_prompt().format(
            question=question,
            gold_answer=gold_answer,
            model_answer=model_answer
        )
        messages = [{"role": "user", "content": prompt}]
        
        response = self.call_model(self.judge_client, self.judge_model, messages,
                                  self.judge_temperature)
        
        if response:
            try:
                # 尝试解析JSON
                response = response.strip()
                if response.startswith("```"):
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]
                result = json.loads(response)
                return float(result["score"]), result.get("reasoning", "")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"    解析评分响应失败: {e}")
                # 尝试提取数字
                import re
                match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', response)
                if match:
                    return float(match.group(1)), ""
        return None

    def _evaluate_one_qa(self, model_config: ModelConfig, qa: Dict, 
                         index: int, total: int) -> Tuple[int, Optional[EvaluationResult]]:
        """评估单个问答对（供多线程调用）。返回 (index, EvaluationResult or None)。"""
        question = qa["question"]
        gold_answer = qa["answer"]
        category = qa.get("category", "UNKNOWN")
        source = qa.get("source_literature", "")

        with self._print_lock:
            print(f"\n[{index+1}/{total}] 问题类别: {category} (模型: {model_config.name})")
            print(f"  问题: {question[:80]}...")

        model_answer = self.get_model_answer(model_config, question)
        if not model_answer:
            with self._print_lock:
                print(f"  - 获取回答失败，跳过")
            return (index, None)
        time.sleep(self.request_delay)

        scores = []
        for round_idx in range(self.num_scoring_rounds):
            result = self.judge_answer(question, gold_answer, model_answer)
            if result:
                score, _ = result
                scores.append(score)
                with self._print_lock:
                    print(f"  [{index+1}/{total}] 轮次{round_idx+1}: {score:.1f}分")
            time.sleep(self.request_delay)

        if not scores:
            return (index, None)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        with self._print_lock:
            print(f"  [{index+1}/{total}] 平均分: {mean_score:.2f} ± {std_score:.2f}")

        return (index, EvaluationResult(
            question=question,
            category=category,
            gold_answer=gold_answer,
            model_answer=model_answer,
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            source_literature=source
        ))

    def evaluate_single_model(self, model_config: ModelConfig, 
                              qa_pairs: List[Dict]) -> Dict:
        """评估单个模型（支持多线程并行评估多个问答对）。"""
        print(f"\n{'='*60}")
        print(f"评估模型: {model_config.name}" + 
              (f" (并行 {self.max_workers} 线程)" if self.max_workers > 1 else " (单线程)"))
        print(f"{'='*60}")

        total = len(qa_pairs)
        results: List[EvaluationResult] = []

        if self.max_workers <= 1:
            # 单线程：保持原有顺序与打印风格
            for i, qa in enumerate(qa_pairs):
                _, ev = self._evaluate_one_qa(model_config, qa, i, total)
                if ev:
                    results.append(ev)
        else:
            # 多线程：并行评估多个问答对
            index_to_result: Dict[int, Optional[EvaluationResult]] = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_one_qa, model_config, qa, i, total): i
                    for i, qa in enumerate(qa_pairs)
                }
                for future in as_completed(futures):
                    idx, ev = future.result()
                    index_to_result[idx] = ev
            # 按原始顺序排列
            for i in range(total):
                if index_to_result.get(i):
                    results.append(index_to_result[i])

        return {
            "model_name": model_config.name,
            "results": results,
            "overall_mean": np.mean([r.mean_score for r in results]) if results else 0,
            "overall_std": np.std([r.mean_score for r in results]) if results else 0,
            "category_scores": self._calculate_category_scores(results)
        }

    def _calculate_category_scores(self, results: List[EvaluationResult]) -> Dict:
        """按类别计算分数"""
        category_scores = {}
        for r in results:
            if r.category not in category_scores:
                category_scores[r.category] = []
            category_scores[r.category].append(r.mean_score)
        
        return {
            cat: {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "count": len(scores)
            }
            for cat, scores in category_scores.items()
        }

    def run_evaluation(self, qa_file: str, output_dir: str = "./eval_results",
                       max_questions: Optional[int] = None,
                       filter_by: Optional[str] = None,
                       filter_threshold: float = 5.0) -> Dict:
        """运行完整评估
        
        Args:
            qa_file: 问答对 JSON 文件
            output_dir: 输出目录
            max_questions: 最大评估问题数
            filter_by: 基准模型已评详细结果 JSON（如 eval_results/details_PollyLLM.json），
                        用于过滤低于 filter_threshold 的问题
            filter_threshold: 过滤阈值（基准模型平均分 < 此值的问题将被剔除）
        """
        # 加载问答对
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        if max_questions:
            qa_pairs = qa_pairs[:max_questions]
        
        total_before = len(qa_pairs)
        
        # 按基准模型得分过滤低质量问题
        if filter_by and os.path.exists(filter_by):
            with open(filter_by, 'r', encoding='utf-8') as f:
                baseline_results = json.load(f)
            # 构建 "问题 → 基准模型平均分" 的映射
            baseline_scores = {
                item["question"]: item["mean_score"]
                for item in baseline_results
                if "question" in item and "mean_score" in item
            }
            before = len(qa_pairs)
            qa_pairs = [
                qa for qa in qa_pairs
                if baseline_scores.get(qa["question"], 999) >= filter_threshold
            ]
            filtered_out = before - len(qa_pairs)
            print(f"[质量过滤] 基准文件: {os.path.basename(filter_by)}")
            print(f"[质量过滤] 阈值: 基准模型平均分 >= {filter_threshold}")
            print(f"[质量过滤] 过滤前: {before} 条 → 过滤后: {len(qa_pairs)} 条（剔除 {filtered_out} 条）")
        
        print(f"加载了 {len(qa_pairs)} 个问答对")
        print(f"待评估模型数量: {len(self.eval_models)}")
        print(f"裁判模型: {self.judge_model}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 评估每个模型
        all_results = {}
        for model_config in self.eval_models:
            model_results = self.evaluate_single_model(model_config, qa_pairs)
            all_results[model_config.name] = model_results
        
        # 保存详细结果
        self._save_detailed_results(all_results, output_dir)
        
        # 生成汇总报告
        summary = self._generate_summary(all_results)
        
        # 保存汇总
        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n汇总报告已保存到: {summary_path}")
        
        # 生成柱形图
        self._generate_charts(summary, output_dir)
        
        return summary

    def _save_detailed_results(self, all_results: Dict, output_dir: str):
        """保存详细评估结果"""
        for model_name, data in all_results.items():
            results_data = []
            for r in data["results"]:
                results_data.append({
                    "question": r.question,
                    "category": r.category,
                    "gold_answer": r.gold_answer,
                    "model_answer": r.model_answer,
                    "scores": r.scores,
                    "mean_score": r.mean_score,
                    "std_score": r.std_score,
                    "source_literature": r.source_literature
                })
            
            safe_name = model_name.replace("/", "_").replace(" ", "_")
            path = os.path.join(output_dir, f"details_{safe_name}.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)

    def _generate_summary(self, all_results: Dict) -> Dict:
        """生成评估汇总"""
        summary = {
            "models": {},
            "ranking": [],
            "category_comparison": {}
        }
        
        # 模型汇总
        for model_name, data in all_results.items():
            summary["models"][model_name] = {
                "overall_mean": data["overall_mean"],
                "overall_std": data["overall_std"],
                "num_questions": len(data["results"]),
                "category_scores": data["category_scores"]
            }
        
        # 排名
        ranking = sorted(
            summary["models"].items(),
            key=lambda x: x[1]["overall_mean"],
            reverse=True
        )
        summary["ranking"] = [
            {"rank": i+1, "model": name, "score": data["overall_mean"], "std": data["overall_std"]}
            for i, (name, data) in enumerate(ranking)
        ]
        
        # 按类别比较
        all_categories = set()
        for data in all_results.values():
            all_categories.update(data["category_scores"].keys())
        
        for cat in all_categories:
            summary["category_comparison"][cat] = {}
            for model_name, data in all_results.items():
                if cat in data["category_scores"]:
                    summary["category_comparison"][cat][model_name] = data["category_scores"][cat]
        
        return summary

    def _generate_charts(self, summary: Dict, output_dir: str):
        """生成评估图表"""
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 总体评分柱形图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = [r["model"] for r in summary["ranking"]]
        scores = [r["score"] for r in summary["ranking"]]
        stds = [r["std"] for r in summary["ranking"]]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.bar(models, scores, yerr=stds, capsize=5, color=colors, edgecolor='black')
        
        ax.set_ylabel('Average Score (0-10)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('Coating LLM Evaluation - Overall Scores', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 10)
        
        # 在柱形上显示分数
        for bar, score, std in zip(bars, scores, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.2,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        chart_path = os.path.join(output_dir, "overall_scores.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"总体评分图已保存到: {chart_path}")
        
        # 2. 按类别的分组柱形图
        if summary["category_comparison"]:
            categories = list(summary["category_comparison"].keys())
            models = list(summary["models"].keys())
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(categories))
            width = 0.8 / len(models)
            
            for i, model in enumerate(models):
                model_scores = []
                model_stds = []
                for cat in categories:
                    if model in summary["category_comparison"].get(cat, {}):
                        model_scores.append(summary["category_comparison"][cat][model]["mean"])
                        model_stds.append(summary["category_comparison"][cat][model]["std"])
                    else:
                        model_scores.append(0)
                        model_stds.append(0)
                
                offset = (i - len(models)/2 + 0.5) * width
                ax.bar(x + offset, model_scores, width, yerr=model_stds, 
                      capsize=3, label=model, alpha=0.8)
            
            ax.set_ylabel('Average Score (0-10)', fontsize=12)
            ax.set_xlabel('Category', fontsize=12)
            ax.set_title('Coating LLM Evaluation - Scores by Category', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 10)
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            
            chart_path = os.path.join(output_dir, "category_scores.png")
            plt.savefig(chart_path, dpi=150)
            plt.close()
            print(f"类别评分图已保存到: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description='涂层LLM评估系统')
    parser.add_argument('-c', '--config', default='eval_config.json',
                       help='评估配置文件路径')
    parser.add_argument('-q', '--qa-file', default='coating_qa_pairs_full_simplified.json',
                       help='问答对JSON文件路径')
    parser.add_argument('-o', '--output-dir', default='./eval_results',
                       help='输出目录')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='最大评估问题数（用于测试）')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='并行线程数（覆盖配置文件；0 或 1 为单线程）')
    parser.add_argument('--filter-by', type=str, default=None,
                       help='基准模型已评详细结果JSON（如 eval_results/details_PollyLLM.json），用于过滤低分问题')
    parser.add_argument('--filter-threshold', type=float, default=5.0,
                       help='过滤阈值：基准模型平均分 < 此值的问题将被剔除（默认 5.0）')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.config)
    if args.max_workers is not None:
        evaluator.max_workers = args.max_workers
    summary = evaluator.run_evaluation(
        qa_file=args.qa_file,
        output_dir=args.output_dir,
        max_questions=args.max_questions,
        filter_by=args.filter_by,
        filter_threshold=args.filter_threshold,
    )
    
    # 打印排名
    print("\n" + "="*60)
    print("评估结果排名")
    print("="*60)
    for r in summary["ranking"]:
        print(f"  #{r['rank']} {r['model']}: {r['score']:.2f} ± {r['std']:.2f}")


if __name__ == "__main__":
    main()
