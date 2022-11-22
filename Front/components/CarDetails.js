import { View, Text, StyleSheet } from "react-native";

function CarDetails({ price, style, textStyle }) {
  return (
    <View style={[styles.details, style]}>
      {/* <Text style={[styles.detailItem, textStyle]}>{brand}</Text> */}
      {/* <Text style={[styles.detailItem, textStyle]}>{name}</Text> */}
      <Text style={[styles.detailItem, textStyle]}>{price}만원</Text>
    </View>
  );
}

export default CarDetails;

const styles = StyleSheet.create({
  details: {
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: 8,
  },
  detailItem: {
    marginHorizontal: 4,
    fontSize: 12,
  },
});
