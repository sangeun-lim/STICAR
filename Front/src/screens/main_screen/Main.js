import React from "react";
import { StyleSheet, Text, View, Image } from "react-native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";

import CameraScreen from "../camera_screen/Camera";
import CategoriesScreen from "../collection/CategoriesScreen";
import Ranking from "../ranking_screen/Ranking";
import Profile from "../profile_screen/Profile";

// 뒤로가기 버튼
import Back_Icon from "../../../assets/images/back_white.png";

function Home() {
  return (
    <View style={styles.container}>
      <Text>Home</Text>
    </View>
  );
}

const Tab = createBottomTabNavigator();

function Main() {
  return (
    <Tab.Navigator
      initialRouteName="Home"
      backBehavior="initialRoute"
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          if (route.name === "Home") {
            iconName = focused
              ? require("../../../assets/images/home_select.png")
              : require("../../../assets/images/home_not_select.png");
          } else if (route.name === "Collection") {
            iconName = focused
              ? require("../../../assets/images/gallery_select.png")
              : require("../../../assets/images/gallery_not_select.png");
          } else if (route.name === "Camera") {
            iconName = focused
              ? require("../../../assets/images/camera_select.png")
              : require("../../../assets/images/camera_not_select.png");
          } else if (route.name === "Ranking") {
            iconName = focused
              ? require("../../../assets/images/ranking_select.png")
              : require("../../../assets/images/ranking_not_select.png");
          } else if (route.name === "Profile") {
            iconName = focused
              ? require("../../../assets/images/profile_select.png")
              : require("../../../assets/images/profile_not_select.png");
          }

          // 여기에 원하는 컴포넌트를 반환 할 수 있습니다!
          return <Image source={iconName} style={{ width: 25, height: 25 }} />;
        },
        tabBarActiveTintColor: "tomato",
        tabBarInactiveTintColor: "gray",
      })}
    >
      <Tab.Screen
        options={{ headerShown: false }}
        name="Camera"
        component={CameraScreen}
      />
      <Tab.Screen
        name="Collection"
        component={CategoriesScreen}
        options={{
          title: "All Categories",
          // headerStyle: { backgroundColor: '#cccccc'},
        }}
      />
      <Tab.Screen name="Home" component={Home} />
      <Tab.Screen name="Ranking" component={Ranking} />
      <Tab.Screen name="Profile" component={Profile} />
    </Tab.Navigator>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  }
});

export default Main;
