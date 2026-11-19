# Day07 · 1과목 7장 — Structured Output과 Pydantic 검증 (A 등급)

> 기능: F2 리뷰 자동 태깅 · 강의안: 노션 Kant > 딥러닝 실전 & 프롬프트 엔지니어링 > 7장 1~3강 ("1강 §5"는 1강의 5번 절)
> 순서: ⓪ 퀴즈 5분 → ① 훑기 25분 → ② 암기 15분 → ③ 구현 50분 → ④ 리뷰 20분 → "마무리"
> 하는 법: 이 파일을 위에서 아래로 따라간다. 강의안은 ①에서 처음 열고, 적힌 곳만 읽는다.

**한 문장 핵심:** LLM 답은 문법 → 구조 → 업무 규칙 순서로 검사하고, 구조는 Pydantic 클래스 하나로 정해 모델에도 넘기고 받은 답도 검사한다.

## ⓪ 시작 퀴즈 (5분)

지난 강의 핵심 코드다. 빈칸을 채우고 `docs/CHEATSHEET.md`로 확인한다.

1. `response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)`
2. 답 문자열 꺼내기: `response.choices[0].message.content`
3. 재현율 = `tp / (tp + fn)`

## ① 훑기 (25분)

각 강의 §13 요약과 아래 📌만 읽는다. **나머지는 소제목만 보고 넘긴다.**

**🎯 핵심 개념**

1. **출력 계약 (JSON Schema)** (1강 §1~3): 모델 답이 지킬 모양(필드·타입·허용값)을 정한 규칙.
2. **Pydantic BaseModel** (2강 §1~3): 출력 계약을 파이썬 클래스로 쓰고, 그 클래스 하나로 스키마 생성과 검증을 함께 한다.
3. **검증 단계** (1강 §2, 3강 §3): 문법 → 구조 → 업무 규칙 순서로 보고, 형식이 통과해도 내용이 맞다는 뜻은 아니다.
4. **strict** (2강 §5): 자동 형 변환("3" → 3)을 막아 LLM 출력의 타입 오류를 드러낸다.
5. **출력 복구** (3강 §1~2): 실패를 지어낸 값으로 메우지 않고, 다시 묻기·사람 검토·중단 중 하나로 보낸다.

**📌 읽을 곳**

- 1강 §8 문제 2: 실패 이유 3개를 답 보기 전에 찾아본다.
- 2강 §5: `"3"`이 `3`으로 바뀌는 예와 `strict=True`.
- 2강 §7 코드: `DocumentAnalysis`가 우리 `ReviewTag`의 본보기다.
- 3강 §2 오류 정책 표, §8 문제 2: `priority=5`를 3으로 잘라 저장하지 않는 이유.

## ② 암기 (15분)

한 번 따라 치고, 가리고 다시 쓰고, 틀린 줄만 반복한다.

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError

class ReviewTag(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")  # 자동 형 변환과 모르는 필드를 막는다
    sentiment: Literal["부정", "중립", "긍정"]                # 허용값은 Literal
    urgency: int = Field(ge=1, le=3)                         # 범위는 Field

try:
    tag = ReviewTag.model_validate_json(raw)                 # 문자열 → 검증된 객체
except ValidationError as error:
    print(error.errors())                                    # 오류마다 loc(위치)·type·msg
```

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError

class ReviewTag(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid)
    sentiment: Literal["부정", "중립", "긍정"]
    urgency: int = [Field(ge=1, le=3)]

try:
    tag = ReviewTag.model_validate_json(raw)                 # 문자열 → 검증된 객체
except ValidationError as error:
    print(error.errors())                                    # 오류마다 loc(위치)·type·ms
```


## ③ 구현 (50분)

> 코드: `labs/day07/dl_ch07_structured_output.py` · 실행: `uv run python labs/day07/dl_ch07_structured_output.py` (② 부분은 Ollama가 켜져 있어야 한다)

외운 코드로 ✍️ 2개를 쓴다. 15분 막히면 정답을 달라고 하고, 다 쓰면 "리뷰해줘"라고 한다. 실행하기 전에 예상을 적는다.

| ✍️ | 목적 | 줄 수 |
|---|---|---|
| `ReviewTag` 필드 | 감성·불만 유형·긴급도·근거 문장 | 4줄 |
| `validate_tag` | 답 문자열 → 성공이면 (객체, []), 실패면 (None, 오류 목록) | 4줄 |

**실행 전 예상**

1. ① 고정된 답 5개 중 "구조 통과"는 몇 개이고, 어느 것일까?
:
2. ② 프롬프트만 줬을 때 리뷰 3개 중 몇 개가 통과할까? 스키마도 주면?
:
3. 스키마를 줬을 때 리뷰 2("박음질이 한 달 만에 풀렸어요")의 불만 유형은 무엇으로 나올까?
:

<details><summary>정답</summary>

```
① [정상] 구조 통과 | 부정/배송/3 | 근거가 리뷰에 있음
   [코드 펜스] 구조 실패 | <root>: json_invalid
   [허용값·범위 밖] 구조 실패 | sentiment: literal_error, urgency: less_than_equal
   [문자열 숫자·추가 필드] 구조 실패 | urgency: int_type, note: extra_forbidden
   [근거 지어냄] 구조 통과 | 부정/배송/3 | 근거가 리뷰에 없음
② 프롬프트만: 0/3 (모두 sentiment 등 missing + 감성·불만 유형·긴급도·근거 문장 extra_forbidden)
   스키마도 전달: 3/3 | 리뷰 1 부정/배송/3, 리뷰 2 부정/배송/3, 리뷰 3 긍정/없음/1
```

1. 2개 (정상, 근거 지어냄). "근거 지어냄"은 구조는 맞지만 근거 문장이 리뷰에 없어 업무 규칙에서 걸린다. 코드 펜스는 JSON 문법부터 틀렸다.
2. 0개 → 3개. 프롬프트만 주면 모델이 키를 한글(감성, 긴급도…)로 지어서 필드 이름이 하나도 안 맞는다. 스키마를 `response_format`으로 넘기면 키 이름과 허용값이 고정된다. (2번 실행해도 같았다)
3. **배송** (틀렸다). 박음질이 풀린 것은 품질·내구성인데, 구조 검증은 허용값 안이기만 하면 통과시킨다.

</details>

## ④ 리뷰 (20분)

**문제 찾기:** 아래 코드는 오류 없이 돌아가지만 실무에서 사고가 나는 곳이 3군데 있다. 찾아서 말해 본다.

```python
class Tag(BaseModel):
    sentiment: str
    urgency: int = Field(ge=1, le=3)


def parse_tag(raw: str) -> Tag:
    try:
        return Tag.model_validate_json(raw)
    except Exception:
        return Tag(sentiment="중립", urgency=1)
```

<details><summary>정답</summary>

1. `sentiment: str`이라 `"매우 부정"`도 통과한다. `Literal`로 허용값을 정한다.
2. `strict`와 `extra` 설정이 없어 `"3"`이 3으로 바뀌고, 모르는 필드는 조용히 버려진다. `ConfigDict(strict=True, extra="forbid")`를 넣는다.
3. 실패하면 가짜 값(중립, 1)을 돌려줘서 실패가 사라진다. `None`이나 오류 목록을 돌려주고, `except Exception`은 `ValidationError`로 좁힌다.

실제로 `parse_tag('{"sentiment": "매우 부정", "urgency": "3"}')`은 `sentiment='매우 부정' urgency=3`을, `parse_tag("죄송하지만 답할 수 없습니다.")`는 `sentiment='중립' urgency=1`을 돌려준다.

</details>

**결과 리뷰:** 스키마를 넘기니 3/3이 통과했다. 이제 태깅이 정확하다고 말해도 되는가? 왜?

<details><summary>모범 답</summary>

아니다. 통과한 것은 모양(구조)과 근거 문장 확인뿐이고, 리뷰 2는 품질 불만인데 "배송"으로 태깅됐다. 구조 검증은 "허용값 중 하나인가"만 보고 "맞는 값인가"는 못 본다. 내용이 맞는지는 Day06처럼 정답을 붙인 평가셋으로 따로 재야 한다 (JSON 문법이 맞다 ≠ Schema를 만족한다 ≠ 내용이 사실이다).

</details>
