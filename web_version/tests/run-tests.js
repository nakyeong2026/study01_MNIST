// 테스트 파일을 모두 불러온 뒤(import 시 test()로 등록됨) 한꺼번에 실행
import { runAll } from "./harness.js";
import "./ops.test.js";
import "./model.test.js";
import "./parity.test.js";
import "./preprocess.test.js";

runAll(document.getElementById("results"));
