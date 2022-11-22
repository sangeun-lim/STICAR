import React, { useState, useEffect } from "react";
import { Button, Image, View, Text, StyleSheet, Platform, TouchableOpacity } from "react-native";
import * as ImagePicker from "expo-image-picker";

import DefaultImage from "../../../assets/profile_images/DefaultImage.png";

function ImagePickerExample() {
  const [image, setImage] = useState(null);

  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    console.log(result);

    if (!result.cancelled) {
      setImage(result.uri);
    }
  };

  return (
      <View style={styles.imageContainer}>
        <TouchableOpacity onPress={pickImage}>
          {!image && <Image source={DefaultImage} style={styles.profile} />}
        </TouchableOpacity>
        <TouchableOpacity onPress={pickImage}>
          {image && <Image source={{ uri: image }} style={styles.profile}/>}
        </TouchableOpacity>
        <Text style={styles.text}>Audi Owner</Text>
        {/* <Button title="Pick an image from camera roll" onPress={pickImage} /> */}
      </View>
  );
}

const styles = StyleSheet.create({
  imageContainer: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    width: "100%"
  },
  profile: {
    width: 80,
    height: 80,
    borderRadius: 100,
    marginHorizontal: 32,
  },
  text: {
    fontSize: 24,
    fontWeight: "bold",
    marginHorizontal: 32,
  }
});

export default ImagePickerExample;