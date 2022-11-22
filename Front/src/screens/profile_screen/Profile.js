import { StyleSheet, Text, View } from "react-native";
import React from "react";

import ProfileImage from './ProfileImage';
import ProfileBadge from "./ProfileBadge";
function Profile() {
  return (
    <View style={styles.container}>
      <ProfileImage></ProfileImage>
      <ProfileBadge></ProfileBadge>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

export default Profile;