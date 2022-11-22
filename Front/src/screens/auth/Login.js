import React, { useState } from "react";
import { StyleSheet, Text, View, Image, useWindowDimensions, ScrollView, TouchableOpacity } from "react-native";
import { useNavigation } from '@react-navigation/native';

import Logo from "../../../assets/images/Car_Logo.jpg";
import CustomInput from "../../../components/ui/CustomInput"; // TextInput 커스텀
import CustomButton from "../../../components/ui/CustomButton"; // Button 커스텀

// 뒤로가기 버튼
import Back_Icon from '../../../assets/images/back_white.png'

const Login = props => {
  const navigation = useNavigation();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const { height } = useWindowDimensions();

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={{ width: '100%', backgroundColor: 'black' }}>
        {/* <Image
          source={Logo}
          style={[styles.logo, { height: height * 0.35 }]}
          resizeMode="contain"
        /> */}
      </View>

      <View style={{ width: '100%', alignItems: 'center', marginVertical: 10, }}>
        {/* <Text style={styles.titletext}>Welcome Back</Text>
        <Text style={styles.bodytext}>Log in to Your Account</Text> */}
      </View>

      <CustomInput
        placeholder="Username" // 입력한 값이 없는 경우 나타나는 배경글
        icon="user" // 입력창의 가장 왼쪽에 배치할 아이콘 이름 (vector-icon 라이브러리 사용)
        value={username} // 입력된 값
        setValue={setUsername} // 입력이 완료된 값
        valueBoolean={username !== '' ? false : true}
      />
      <CustomInput
        placeholder="Password"
        icon="lock"
        value={password}
        setValue={setPassword}
        secureTextEntry={true} // 입력하면 * 으로 표기
        valueBoolean={password !== '' ? false : true}
      />

      <View style={{ width: "90%" }}>
        <Text
          style={[styles.bodytext, { alignSelf: 'flex-end', color: 'gray', marginVertical: 5, }]}
          onPress={() => navigation.navigate("FindPassword")}
        >Forgot Password?</Text>
      </View>

      <CustomButton
        text="로그인"
        type="Login"
        onPress={() => navigation.navigate("Main")}
      />

      <TouchableOpacity
        style={styles.back_icon}
        onPress={() => navigation.navigate("Begin")}>
        <Image
          style={styles.back_icon_image}
          source={Back_Icon} />
      </TouchableOpacity>
    </ScrollView >
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    width: "100%",
    alignItems: "center",
    justifyContent: 'center',
    backgroundColor: '#ADC0D9',
  },
  logo: {
    width: "100%",
  },
  titletext: {
    fontSize: 40,
    fontStyle: 'italic',
    marginVertical: 10,
  },
  bodytext: {
    fontSize: 12,
    fontStyle: 'bold',
    marginVertical: 5,
  },
  back_icon: {
    position: 'absolute',
    top: "5%",
    left: "5%",
    width: '10%',
    maxWidth: 60,
    aspectRatio: 1,
    maxHeight: 60,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 3,
  },
  back_icon_image: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  }
});

export default Login;
