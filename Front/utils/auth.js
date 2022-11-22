import axios from "axios";

async function authenticate(mode, email, password) {
  const url = `http://j7c102.p.ssafy.io:8080/${mode}`;

  const response = await axios.post(url, {
    email: email,
    password: password,
  });
}

export async function createUser(email, password) {
  await authenticate("signup", email, password);
  // headers: {
  //   "Content-Type": "application/json",
  // },
}

export async function login(email, password) {
  await authenticate("login", email, password);
}
