import { FlatList, StyleSheet } from "react-native";
import CategoryGridTile from "../../../components/CategoryGridTile";

import { CATEGORIES } from "../../../data/CarData";

function CategoriesScreen({ navigation }) {
  function renderCategoryItem(itemData) {
    function pressHandler() {
      navigation.navigate("CarsOverview", {
        categoryId: itemData.item.id,
      });
    }

    return (
      <CategoryGridTile
        title={itemData.item.title}
        img={itemData.item.img}
        onPress={pressHandler}
      />
    );
  }
  return (
    <FlatList
      data={CATEGORIES}
      keyExtractor={(item) => item.id}
      renderItem={renderCategoryItem}
      numColumns={2}
    />
  );
}

export default CategoriesScreen;

const styles = StyleSheet.create({
  image: {
    width: "100%",
    height: 200,
  }
})