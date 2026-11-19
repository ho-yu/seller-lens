# Day08 · 1과목 8장 — Tool Calling (A 등급)

> 기능: F5 리포트 v0 · 강의안: 노션 Kant > 딥러닝 실전 & 프롬프트 엔지니어링 > 8장 1~3강 ("2강 §4"는 2강의 4번 절)
> 끝의 기준은 하나다. 아래 질문에 내 말로 답할 수 있고 ✍️ 두 곳을 쓸 수 있으면 끝이다. 시간표는 없다.
> 모르는 것이 나오면 강의안을 뒤지지 말고 Claude에게 한 줄로 묻는다.

## 오늘의 질문

**모델이 "도구를 호출한다"고 할 때 함수를 실제로 실행하는 것은 누구이고, 실행하기 전에 내 코드는 무엇을 확인해야 하는가?**

지금은 답을 몰라도 된다. 코드를 돌려 본 뒤에 답한다.

## 🎯 핵심 개념 (한 번만 읽고 넘어간다)

1. **Tool Calling** (1강 §1~2): 모델은 "이 도구를 이 인자로 부르자"는 제안(도구 이름 + 인자 JSON 문자열)만 만들고, 함수는 내 코드가 실행한다.
2. **실행 전 확인** (2강 §2~4): 모델이 만든 인자는 믿을 수 없는 외부 입력이다. 등록한 도구인지 보고(allowlist), 인자를 Pydantic으로 검증한 뒤에만 실행한다.
3. **결과 돌려주기** (3강 §1~4): 실행 결과를 호출 id와 함께 `role="tool"` 메시지로 붙여 다시 물으면 모델이 최종 답을 쓴다. 실패도 `ok: false`로 그대로 알린다.

## 코드 먼저

> 코드: `labs/day08/dl_ch08_tool_calling.py` · 실행: `uv run python labs/day08/dl_ch08_tool_calling.py` (② 부분은 Ollama가 켜져 있어야 한다)

아래 🧠 코드를 **보면서** ✍️ 두 곳을 쓰고, 실행해서 출력을 본다. 막히면 바로 물어본다.

| ✍️ | 목적 | 줄 수 |
|---|---|---|
| `run_tool` | 모델이 제안한 (도구 이름, 인자 문자열)을 확인하고 실행한다 | 7줄 (가운데 `try` 4줄은 Day07과 같다) |
| `tool_messages` | 도구 호출마다 실행 결과를 `role="tool"` 메시지로 만든다 | 5줄 |

```python
TOOLS = [{"type": "function", "function": {"name": "count_reviews", "description": "...", "parameters": CountArgs.model_json_schema()}}]
reply = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS).choices[0].message

def run_tool(name: str, raw_arguments: str) -> dict:
    if name not in TOOL_ROUTER:                              # 등록한 도구만 실행한다 (allowlist)
        return {"ok": False, "error_code": "unknown_tool"}
    try:
        args = CountArgs.model_validate_json(raw_arguments)  # 모델이 만든 인자를 검증한다
    except ValidationError:
        return {"ok": False, "error_code": "invalid_arguments"}
    return TOOL_ROUTER[name](**args.model_dump())            # 통과한 것만 내 코드가 실행한다

for call in reply.tool_calls:                                # 모델의 제안: 이름, 인자 문자열, 호출 id
    result = run_tool(call.function.name, call.function.arguments)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
```

<details><summary>실행 결과 (내 출력과 비교)</summary>

```
① 모델 없이: 고정된 도구 호출 4개
  [정상] {'ok': True, 'product_id': 'product_03', 'sentiment': '부정', 'count': 27, 'total': 64}
  [없는 도구] {'ok': False, 'error_code': 'unknown_tool'}
  [인자 틀림] {'ok': False, 'error_code': 'invalid_arguments'}
  [없는 상품] {'ok': False, 'error_code': 'product_not_found'}
  tool 메시지: [{'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"ok": true, "product_id": "product_03", "sentiment": "부정", "count": 27, "total": 64}'}]

② Ollama: 도구 없이
  Q. product_03의 부정 리뷰는 몇 건이야?
  A. 도구 결과가 제공되지 않았기 때문에, product_03의 부정 리뷰 건수를 확인할 수 없습니다. (뒤 생략. product_99도 같은 답)

② Ollama: 도구와 함께
  Q. product_03의 부정 리뷰는 몇 건이야?
    모델의 제안: count_reviews {"product_id":"product_03","sentiment":"부정"}
    내 코드의 실행 결과: {"ok": true, "product_id": "product_03", "sentiment": "부정", "count": 27, "total": 64}
  A. product_03의 부정 리뷰는 27건입니다.
  Q. product_99의 부정 리뷰는 몇 건이야?
    모델의 제안: count_reviews {"product_id":"product_99","sentiment":"부정"}
    내 코드의 실행 결과: {"ok": false, "error_code": "product_not_found"}
  A. product_99은 존재하지 않아서 부정 리뷰 건수를 확인할 수 없습니다.
  Q. 부정 리뷰에는 어떻게 답글을 달면 좋을까? 한 문장으로.
  A. (모델의 제안 없이 바로 답한다. 문장은 실행할 때마다 조금 다르다)
```

2번 실행해 같았다 (답글 문장만 조금 다르다).

- 도구 없이는 모델이 건수를 알 길이 없다. 도구와 함께면 모델이 `count_reviews`를 제안하고, 27이라는 숫자는 내 코드(pandas)가 만든다.
- product_99는 인자 형식은 통과했지만 함수 안에서 `product_not_found`로 걸렸다. 실패를 그대로 돌려주니 모델도 숫자를 지어내지 않았다.
- 답글 질문에는 모델이 도구를 제안하지 않았다. 도구를 쓸지 말지는 모델이 정하고, 쓰겠다고 한 것을 실행할지는 내 코드가 정한다.

</details>

## 내 말로 답하기

🎯 3줄을 다시 읽고, 오늘의 질문에 두세 문장으로 답해서 Claude에게 보낸다. Claude가 빈틈만 짚어 준다.

<details><summary>모범 답 (내 답을 쓴 뒤에 연다)</summary>

함수를 실행하는 것은 모델이 아니라 내 코드다. 모델은 도구 이름과 인자를 JSON 문자열로 제안할 뿐이다. 내 코드는 등록한 도구 이름인지, 인자가 `CountArgs`에 맞는지 확인한 뒤에 실행하고, 함수 안에서는 데이터에 실제로 있는 상품인지도 본다. 결과는 성공이든 실패든 그대로 `role="tool"` 메시지로 돌려주고, 건수는 모델이 아니라 pandas가 센다.

</details>

## 더 하고 싶을 때 (선택)

안 해도 된다. 궁금한 것만 고른다.

- **실행 전 예상:** ①의 호출 4개 중 `count_reviews` 함수가 실제로 실행되는 것은 몇 개일까? (답: 2개. 정상과 없는 상품. 없는 도구와 인자 틀림은 함수에 닿기 전에 막힌다)
- **문제 찾기:** 아래 코드는 오류 없이 돌아가지만 사고가 나는 곳이 3군데 있다.

  ```python
  def delete_reviews(product_id: str) -> dict:  # 관리자만 쓰는 함수. 도구로 등록한 적 없다
      return {"ok": True, "deleted": product_id}

  def run_tool(name: str, raw_arguments: str) -> dict:
      try:
          function = globals()[name]
          return function(**json.loads(raw_arguments))
      except Exception:
          return {"ok": True, "count": 0}
  ```

  <details><summary>정답</summary>

  1. `globals()[name]`은 파일 안의 모든 함수를 열어 준다. `run_tool("delete_reviews", '{"product_id": "product_03"}')`가 `{'ok': True, 'deleted': 'product_03'}`을 돌려준다. `TOOL_ROUTER`에 등록한 이름만 실행한다.
  2. `json.loads`는 문법만 본다. `"sentiment": "나쁨"` 같은 값이 그대로 함수에 들어간다. `CountArgs.model_validate_json`으로 검증한다.
  3. 실패를 가짜 성공으로 메운다. `"sentiment": "나쁨"`으로 부르면 `{'ok': True, 'count': 0}`이 나와서 모델은 "0건"이라고 답한다. `ok: False`와 `error_code`를 돌려준다.

  </details>

- **📌 강의안에서 읽을 곳:** 1강 §2 "먼저 확인할 실행 경계" 그림, 2강 §2 검증 층 표, 3강 §4 성공과 실패 응답 정책. 강의 코드는 유료 OpenAI Responses API라서 이름이 다르다: `function_call` item은 우리의 `tool_calls`, `call_id`는 `call.id`, `function_call_output`은 `role="tool"` 메시지다.
- **빈칸 퀴즈** (`docs/CHEATSHEET.md`로 확인): ① 문자열을 검증된 객체로: `tag = ReviewTag.______(raw)` ② 자동 형 변환과 모르는 필드 막기: `ConfigDict(strict=True, extra="______")` ③ 답 문자열 꺼내기: `response.______[0].message.content`
