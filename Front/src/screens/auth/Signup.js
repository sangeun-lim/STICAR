import React, { useState } from "react";
import { StyleSheet, Text, View, TouchableOpacity, Image, ScrollView } from "react-native";
import { useNavigation } from '@react-navigation/native';

import CustomInput from "../../../components/ui/CustomInput";
import CustomButton from "../../../components/ui/CustomButton";

// 뒤로가기 버튼
import Back_Icon from '../../../assets/images/back_white.png'

function SignUp({ navigation }) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [repeatpassword, setRepeatPassword] = useState("");

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Create an Account</Text>

      <View style={{ width: "100%", marginVertical: 50, alignItems: 'center', justifyContent: 'space-evenly' }}>
        <CustomInput
          placeholder="Username"
          icon="user"
          value={username}
          setValue={setUsername}
          valueBoolean={username !== '' ? false : true}
        />
        <CustomInput
          placeholder="Email"
          icon="envelope-o"
          value={email}
          setValue={setEmail}
          valueBoolean={email !== '' ? false : true}
        />
        <CustomInput
          placeholder="Password"
          icon="lock"
          value={password}
          setValue={setPassword}
          secureTextEntry={true}
          valueBoolean={password !== '' ? false : true}
        />
        <CustomInput
          placeholder="Repeat Password"
          icon="lock"
          value={repeatpassword}
          setValue={setRepeatPassword}
          secureTextEntry={true}
          valueBoolean={password !== repeatpassword ? true : false}
        />
      </View>

      <CustomButton
        text="회원가입"
        type="SignUp"
        onPress={() => navigation.navigate("Login")}
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
    justifyContent: 'flex-start',
    backgroundColor: '#ADC0D9',
    paddingVertical: 20
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    fontStyle: 'italic',
    marginVertical: 20,
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
  },
});

export default SignUp;
