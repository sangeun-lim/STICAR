// 제조사, 차종명, 이미지, 가격, 종류(세단인지 suv인지), 연비
class Car {
  constructor(id, brand, name, imageUrl, price, type, fuelEfficiency) {
    this.id = id;
    this.brand = brand;
    this.name = name;
    this.imageUrl = imageUrl;
    this.price = price;
    this.type = type;
    this.fuelEfficiency = fuelEfficiency;
  }
}

export default Car;
