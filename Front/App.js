import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";

import begin_screen from "./src/screens/Begin";
import login from "./src/screens/auth/Login";
import signup from "./src/screens/auth/Signup";
import findpassword_screen from "./src/screens/auth/FindPassword";
import main_screen from "./src/screens/main_screen/Main";
import CarsOverview from "./src/screens/collection/CarsOverview";
// import CategoriesScreen from "./src/screens/collection/CategoriesScreen";
import CarDetailScreen from "./src/screens/collection/CarDetailScreen";
import SignupScreen from "./src/screens/auth/SignupScreen";
import LoginScreen from "./src/screens/auth/LoginScreen";

import { StatusBar } from "expo-status-bar";
import { Colors } from "./constants/Colors";

const Stack = createStackNavigator();

// function AuthStack() {
//   return (
//     <Stack.Navigator
//       screenOptions={{
//         headerStyle: { backgroundColor: Colors.primary500 },
//         headerTintColor: "white",
//         contentStyle: { backgroundColor: Colors.primary100 },
//       }}
//     >
//       <Stack.Screen name="LoginScreen" component={LoginScreen} />
//       <Stack.Screen name="SignupScreen" component={SignupScreen} />
//       <Stack.Screen name="Begin" component={begin_screen} />
//       <Stack.Screen name="Login" component={login} />
//       <Stack.Screen name="SignUp" component={signup} />
//       <Stack.Screen name="FindPassword" component={findpassword_screen} />
//       <Stack.Screen name="Main" component={main_screen} />
//       {/* <Stack.Screen name="CategoriesScreen" component={CategoriesScreen} /> */}
//       <Stack.Screen name="CarsOverview" component={CarsOverview} />
//       <Stack.Screen name="CarDetail" component={CarDetailScreen} />
//     </Stack.Navigator>
//   );
// }

// function AuthenticatedStack() {
//   return (
//     <Stack.Navigator
//       screenOptions={{
//         headerStyle: { backgroundColor: Colors.primary500 },
//         headerTintColor: "white",
//         contentStyle: { backgroundColor: Colors.primary100 },
//       }}
//     >
//       <Stack.Screen name="Main" component={main_screen} />
//     </Stack.Navigator>
//   );
// }

// function Navigation() {
//   return (
//     <NavigationContainer>
//       <AuthStack />
//     </NavigationContainer>
//   );
// }

function App() {
  return (
    <>
      <StatusBar style="dark" />
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Begin" >
          {/* <Stack.Navigator screenOptions={{ headerShown: false }}> */}
          {/* <Navigation /> */}
          <Stack.Screen options={{ headerShown: false }} name="Begin" component={begin_screen} />
          <Stack.Screen options={{ headerShown: false }} name="Login" component={login} />
          <Stack.Screen options={{ headerShown: false }} name="SignUp" component={signup} />
          <Stack.Screen name="FindPassword" component={findpassword_screen} />
          <Stack.Screen options={{ headerShown: false }} name="Main" component={main_screen} />
          {/* <Stack.Screen name="CategoriesScreen" component={CategoriesScreen} /> */}
          <Stack.Screen name="CarsOverview" component={CarsOverview} />
          <Stack.Screen name="CarDetail" component={CarDetailScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F9FBFC",
    alignItems: "center",
    justifyContent: "center",
  },
});

export default App;
