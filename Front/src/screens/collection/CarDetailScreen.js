import { Image, Text, View, StyleSheet, ScrollView } from "react-native";
import List from "../../../components/CarDetail/List";
import Subtitle from "../../../components/CarDetail/Subtitle";
import CarDetails from "../../../components/CarDetails";

import { CARS } from "../../../data/CarData";

function CarDetailScreen({ route }) {
  const carId = route.params.carId;

  const selectCar = CARS.find((car) => car.id === carId);

  return (
    <ScrollView style={styles.rootContainer}>
      <Image style={styles.image} source={{ uri: selectCar.imageUrl }} />
      <Text style={styles.title}>{selectCar.name}</Text>
      <CarDetails
        brand={selectCar.brand}
        name={selectCar.name}
        price={selectCar.price}
        textStyle={styles.detailText}
      />
      <View style={styles.listOuterContainer}>
        <View style={styles.listContainer}>
          <Subtitle>타입</Subtitle>
          <List data={selectCar.type} />
          <Subtitle>연비</Subtitle>
          <List data={selectCar.fuelEfficiency} />
        </View>
      </View>
    </ScrollView>
  );
}

export default CarDetailScreen;

const styles = StyleSheet.create({
  rootContainer: {
    marginBottom: 32,
  },
  image: {
    width: "100%",
    height: 200,
  },
  title: {
    fontWeight: "bold",
    fontSize: 24,
    margin: 8,
    textAlign: "center",
    color: "grey",
  },
  detailText: {
    color: "red",
  },
  listOuterContainer: {
    alignItems: "center",
  },
  listContainer: {
    width: "80%",
  },
});
