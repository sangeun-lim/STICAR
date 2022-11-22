import { View, Text, StyleSheet } from "react-native";
import React from "react";

function FindPassword() {
  return (
    <View style={styles.container}>
      <Text>findpassword</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
});

export default FindPassword;
