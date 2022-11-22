import React, { Component } from "react";
import { StyleSheet, Text, View, Image, useWindowDimensions } from "react-native";

import Init_Car_Logo from "../../assets/images/Sticar_Begin_Logo.png";
import CustomButton from "../../components/ui/CustomButton";

export default function Begin({ navigation }) {
  return (
    <View style={styles.container}>
      <View style={styles.text_container}>
        <Text style={styles.guide_text}>이 자동차 무엇일까요?</Text>
        <Text style={styles.guide_text}>궁금증을 해결해 드립니다.</Text>
        <Text style={styles.guide_text}>지금 시작해보세요. StiCar</Text>
      </View>

      <Image
        source={Init_Car_Logo}
        style={styles.init_car_logo}
        resizeMode="contain"
      />

      <View style={styles.button_container}>
        <CustomButton
          text="회원가입"
          type="SignUp"
          onPress={() => navigation.navigate("SignUp")}
        />
        <CustomButton
          text="로그인"
          type="Login"
          onPress={() => navigation.navigate("Login")}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    width: "100%",
    height: "100%",
    backgroundColor: 'lightgreen',
    alignItems: 'center',
    justifyContent: 'center',
  },
  text_container: {
    flex: 1,
    top: 100,
    left: 45,
    width: "100%",
    justifyContent: 'flex-start',
    alignItems: 'flex-start',
  },
  guide_text: {
    fontSize: 23,
    fontStyle: 'italic',
    fontWeight: 'bold',
  },
  init_car_logo: {
    position: 'absolute',
    width: "100%",
    maxWidth: 300,
  },
  button_container: {
    flex: 1,
    bottom: 30,
    width: "100%",
    alignItems: "center",
    justifyContent: "flex-end",
  },
});
