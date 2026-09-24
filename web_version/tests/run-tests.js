// 테스트 파일을 모두 불러온 뒤(import 시 test()로 등록됨) 한꺼번에 실행
import { test, runAll } from "./harness.js";

const TEST_MODULES = ["./ops.test.js", "./model.test.js", "./parity.test.js", "./preprocess.test.js"];

async function loadAll() {
  for (const path of TEST_MODULES) {
    try {
      await import(path);
    } catch (err) {
      // 모듈 자체를 불러오지 못해도(문법 오류, 잘못된 경로 등) 실패로 기록해서
      // runAll이 계속 실행되고 제목이 FAIL k/n이 되도록 함
      test(`${path} 불러오기`, () => {
        throw err;
      });
    }
  }
}

await loadAll();
runAll(document.getElementById("results"));
