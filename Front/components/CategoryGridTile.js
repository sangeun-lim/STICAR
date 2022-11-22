import {
  Pressable,
  Text,
  View,
  StyleSheet,
  Platform,
  Image,
} from "react-native";
// import { useNavigation } from "@react-navigation/native";

function CategoryGridTile({ title, img, onPress }) {
  // const navigation = useNavigation();

  return (
    <View style={styles.gridItem}>
      {/* 그리드타일을 눌렀을때 새로운 페이지로 이동하기 위한 Pressable */}
      <Pressable
        android_ripple={{ color: "#ccc" }}
        style={({ pressed }) => [
          styles.button,
          pressed ? styles.buttonPressed : null,
        ]}
        onPress={onPress}
      >
        <View style={styles.innerContainer}>
          <Image source={{ uri: img }} style={styles.bgImage} />
          <Text style={styles.title}>{title}</Text>
        </View>
      </Pressable>
    </View>
  );
}

export default CategoryGridTile;

const styles = StyleSheet.create({
  gridItem: {
    flex: 1,
    margin: 16,
    height: 150,
    borderRadius: 8,
    elevation: 4,
    backgroundColor: "white",
    shadowColor: "black",
    shadowOpacity: 0.25,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 8,
    overflow: Platform.OS === "android" ? "hidden" : "visible",
  },
  button: {
    flex: 1,
  },
  buttonPressed: {
    opacity: 0.5,
  },
  innerContainer: {
    flex: 1,
    padding: 0,
    borderRadius: 8,
    justifyContent: "center",
    alignItems: "center",
    overflow: "hidden",
  },
  bgImage: {
    resizeMode: "contain",
    width: 150,
    height: 100,
  },
  title: {
    fontWeight: "bold",
    fontSize: 18,
  },
});
