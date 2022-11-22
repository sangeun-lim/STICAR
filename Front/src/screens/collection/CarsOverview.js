import { useLayoutEffect } from "react";
import { View, StyleSheet, FlatList } from "react-native";
import CarItem from "../../../components/CarItem";
// import { useRoute } from "@react-navigation/native";

import { CARS, CATEGORIES } from "../../../data/CarData";

function CarsOverview({ route, navigation }) {
  // const route = useRoute();
  const cateId = route.params.categoryId;

  const displayedCars = CARS.filter((carItem) => {
    return carItem.brand.indexOf(cateId) >= 0;
  });

  useLayoutEffect(() => {
    const categoryTitle = CATEGORIES.find(
      (category) => category.id === cateId
    ).title;

    navigation.setOptions({
      name: categoryTitle,
    });
  }, [cateId, navigation]);

  function renderCarItem(itemData) {
    const item = itemData.item;

    const carItemProps = {
      id: item.id,
      brand: item.brand,
      name: item.name,
      imageUrl: item.imageUrl,
      price: item.price,
      type: item.type,
      fuelEfficiency: item.fuelEfficiency,
    };
    return <CarItem {...carItemProps} />;
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={displayedCars}
        keyExtractor={(item) => item.id}
        renderItem={renderCarItem}
      />
    </View>
  );
}

export default CarsOverview;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
});
