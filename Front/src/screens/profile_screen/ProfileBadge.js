import { Button, Image, View, Text, StyleSheet, Platform } from "react-native";

function ProfileBadge() {
  return (
    <View style={styles.container}>
      <View style={styles.horizon}/>
      <Text style={styles.text}>Badge</Text>
    </View>
  )
};
const styles = StyleSheet.create({
  container: {
    flex: 2,
  },
  text: {
    fontSize: 24,
    fontWeight: "bold",
    padding: 32,
  },
  horizon: {
    borderBottomColor: '#A2A2A2',
    borderBottomWidth: 1,
  },
});
export default ProfileBadge;