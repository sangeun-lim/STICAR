import axios from 'axios';
const API = axios.create();

export const DevelopList = () => API.get("/api/developer/"); // 개발자 리스트 출력
export const DevelopCreate = ((name, position, age) => API.post("/api/developer/", {
  name: name,
  position: position,
  age: age
}));