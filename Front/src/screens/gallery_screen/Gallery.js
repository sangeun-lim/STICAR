import { StyleSheet, Text, View } from "react-native";
import React from "react";

function Gallery() {
  return (
    <View style={styles.container}>
      <Text>gallery</Text>
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

export default Gallery;
