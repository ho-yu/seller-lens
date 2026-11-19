"""F2 리뷰 태깅의 출력 계약과 검증 함수. Day07 실습(`labs/day07/`)에서 직접 쓴 것을 옮겨 왔다."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ComplaintType = Literal[
    "배송", "파손", "품질·내구성", "사이즈·호환", "사용성", "소재·냄새", "가격", "CS", "없음"
]


class ReviewTag(BaseModel):
    """리뷰 태깅 결과의 출력 계약. 정의 안 한 필드(extra)와 자동 형 변환(strict)은 막는다 (Day07)."""

    model_config = ConfigDict(strict=True, extra="forbid")
    sentiment: Literal["부정", "중립", "긍정"]
    complaint_type: ComplaintType
    urgency: int = Field(ge=1, le=3)
    evidence: str = Field(min_length=1)


def short_errors(error: ValidationError) -> list[str]:
    """ValidationError를 '필드: 오류 종류' 목록으로 줄인다. 필드가 없는 오류(JSON 문법 등)는 <root>."""
    return [f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['type']}" for e in error.errors()]


def validate_tag(raw: str) -> tuple[ReviewTag | None, list[str]]:
    """답 문자열을 ReviewTag로 검증한다. 성공하면 (객체, []), 실패하면 (None, 오류 목록) (Day07)."""
    try:
        tag = ReviewTag.model_validate_json(raw)  # 문자열 → 검증된 객체
    except ValidationError as error:
        return None, short_errors(error)  # 실패: 객체 없이 오류 목록만
    return tag, []  # 성공: 객체와 빈 목록


def evidence_in_review(tag: ReviewTag, review: str) -> bool:
    """업무 규칙: 근거 문장이 리뷰 원문에 실제로 있는가. 구조 검증으로는 못 잡는다 (Day07)."""
    return tag.evidence.strip() in review
