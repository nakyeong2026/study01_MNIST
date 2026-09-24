// 외부 라이브러리 없이 브라우저에서 돌리는 아주 작은 테스트 하네스

const tests = [];

/** 테스트 등록 (비동기 함수도 가능) */
export function test(name, fn) {
  tests.push({ name, fn });
}

export function assert(condition, message = "assert 실패") {
  if (!condition) throw new Error(message);
}

export function assertEqual(actual, expected, message = "") {
  if (actual !== expected) {
    throw new Error(`${message} 기대값 ${expected}, 실제값 ${actual}`);
  }
}

/** 숫자 또는 숫자 배열이 허용 오차 안에서 같은지 확인 */
export function assertClose(actual, expected, tol = 1e-6, label = "") {
  const a = typeof actual === "number" ? [actual] : Array.from(actual);
  const e = typeof expected === "number" ? [expected] : Array.from(expected);
  if (a.length !== e.length) {
    throw new Error(`${label} 길이가 다름: ${a.length} != ${e.length}`);
  }
  for (let i = 0; i < a.length; i++) {
    // NaN도 실패로 잡기 위해 !(차이 <= 오차) 형태로 비교
    if (!(Math.abs(a[i] - e[i]) <= tol)) {
      throw new Error(`${label} [${i}] ${a[i]} != ${e[i]} (허용 오차 ${tol})`);
    }
  }
}

/** 등록된 테스트를 모두 실행하고 결과를 목록과 문서 제목에 표시 */
export async function runAll(listEl) {
  let passed = 0;
  for (const { name, fn } of tests) {
    const item = document.createElement("li");
    try {
      await fn();
      passed++;
      item.textContent = `PASS  ${name}`;
      item.className = "pass";
    } catch (err) {
      item.textContent = `FAIL  ${name} — ${err.message}`;
      item.className = "fail";
      console.error(name, err);
    }
    listEl.appendChild(item);
  }
  const summary = `${passed === tests.length ? "PASS" : "FAIL"} ${passed}/${tests.length}`;
  document.title = summary;
  document.getElementById("summary").textContent = summary;
}
