import axios from "axios";

const API = axios.create({
  baseURL: "http://j7c102.p.ssafy.io:8080/",
  headers: {
    "Content-Type": "application/json",
  },
});
