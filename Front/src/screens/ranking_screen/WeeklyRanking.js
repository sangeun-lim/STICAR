import { StyleSheet, Text, View } from "react-native";
import React from "react";

function WeeklyRanking() {
  return (
    <View style={styles.container}>
      <Text>Weekly</Text>
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

export default WeeklyRanking;
