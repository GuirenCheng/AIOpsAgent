"""诊断报告「根因结论」硬校验器。

报告生成后，把报告里的事实性结论逐条拿去跟执行证据核对：
有证据支撑的标「✅ 有据可查」，挂不上的标「⚠️ 无据可查」，
让推理幻觉显式暴露、供人工复核，而不是默默流进报告。

复用 replanner 里传入的同一个 llm（temperature=0）做结构化输出；
任何异常都 fail-open（返回空 verdicts），绝不因校验失败而丢报告。
"""
from textwrap import dedent
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field


class ClaimVerdict(BaseModel):
    """单条事实性结论的校验结果。"""

    claim: str = Field(..., description="报告中的一条事实性结论（尽量原文引用一句话）")
    grounded: bool = Field(..., description="该结论是否能从执行证据中得到支撑")
    evidence_ref: str = Field("", description="支撑该结论的证据摘录（grounded=true 时填）；无据可查时为空")


class VerificationResult(BaseModel):
    """逐条校验结果。"""

    verdicts: List[ClaimVerdict] = Field(default_factory=list, description="对报告中事实性结论的逐条校验")


VERIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是一个诊断报告事实核对员。下面给出：
                1. 一份已生成的诊断报告；
                2. 该报告的「执行历史/证据」（来自工具查询的真实结果）。

                请从报告中抽取「事实性结论」（尤其是根因结论、关键发现、处理建议等断言性内容），
                逐条判断它是否真的能从执行历史/证据中得到支撑：
                - grounded=true：证据里确实能查到对应的数据/现象；
                - grounded=false：报告里写了，但执行历史/证据里找不到依据，或纯属推测。

                注意：
                - 只抽取「事实性结论」，不抽取模板化的标题、客套话、过程描述。
                - claim 尽量原文引用一句话，便于定位。
                - grounded=true 时在 evidence_ref 里给出支撑它的证据摘录；false 时留空。
                - 只输出 verdicts 列表，不要输出 JSON 以外的内容。
            """).strip(),
        ),
        ("human", "报告：\n{report}\n\n执行历史/证据：\n{execution_history}"),
    ]
)


async def verify_report(report: str, execution_history: str, llm) -> VerificationResult:
    """LLM 逐条核对报告结论是否有证据支撑；任何异常兜底返回空 verdicts（不仲裁、不破坏报告）。"""
    try:
        chain = VERIFY_PROMPT | llm.with_structured_output(VerificationResult)
        result = await chain.ainvoke({"report": report, "execution_history": execution_history})

        if isinstance(result, VerificationResult):
            return result
        if isinstance(result, dict):
            return VerificationResult(**result)
        logger.warning(f"报告校验返回异常类型 {type(result)}，跳过校验")
        return VerificationResult(verdicts=[])
    except Exception as e:  # noqa: BLE001 —— 校验失败一律兜底，不影响报告输出
        logger.warning(f"报告可信度校验失败，跳过：{e}")
        return VerificationResult(verdicts=[])


def build_verification_section(result: VerificationResult) -> str:
    """把校验结果渲染成报告末尾的 Markdown 小节；无 verdicts 时返回空串。"""
    verdicts = result.verdicts
    if not verdicts:
        return ""

    ungrounded = [v for v in verdicts if not v.grounded]
    lines = ["## 🛡️ 结论可信度校验", ""]
    for v in verdicts:
        if v.grounded:
            ref = f"（依据：{v.evidence_ref}）" if v.evidence_ref else ""
            lines.append(f"- ✅ 有据可查：{v.claim}{ref}")
        else:
            lines.append(f"- ⚠️ 无据可查：{v.claim}")
    lines.append("")
    lines.append(
        f"> 共校验 {len(verdicts)} 条结论，{len(ungrounded)} 条无据可查；无据的结论请在采纳前人工复核。"
    )
    return "\n".join(lines)
