import Category from "../models/Category";
import CAR from "../models/Car";

export const CATEGORIES = [
  new Category(
    "c1",
    "Audi",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/audi.png"
  ),
  new Category(
    "c2",
    "Benz",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/benz.png"
  ),
  new Category(
    "c3",
    "BMW",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/bmw.png"
  ),
  new Category(
    "c4",
    "Chevrolet",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/chevrolet.png"
  ),
  new Category(
    "c5",
    "Ford",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/ford.png"
  ),
  new Category(
    "c6",
    "Genesis",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/genesis.png"
  ),
  new Category(
    "c7",
    "Hyundai",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/hyundai.png"
  ),
  new Category(
    "c8",
    "Kia",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/kia.png"
  ),
  new Category(
    "c9",
    "LandRover",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/landrover.png"
  ),
  new Category(
    "c10",
    "RenaultSamsung",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/renault.png"
  ),
  new Category(
    "c11",
    "SsangYong",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/ssangyong.png"
  ),
  new Category(
    "c12",
    "Volkswagen",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/brand/Volkswagen.png"
  ),
];

export const CARS = [
  new CAR(
    "m1",
    ["c1"],
    "A6",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/audi/a6.png",
    "6744~8824",
    ["준대형 세단"],
    ["10.8~15"]
  ),

  new CAR(
    "m2",
    ["c2"],
    "C Class",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/benz/c+class.png",
    "6150~6850",
    ["중형 세단"],
    ["11.3~11.8"]
  ),
  new CAR(
    "m3",
    ["c2"],
    "CLS Class",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/benz/cls+class.png",
    "9370~11410",
    ["준대형 쿠페"],
    ["9.1~13.9"]
  ),
  new CAR(
    "m4",
    ["c2"],
    "S Class",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/benz/s+class.png",
    "14060~23060",
    ["대형 세단"],
    ["7.5~12"]
  ),
  new CAR(
    "m5",
    ["c2"],
    "GLC Class",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/benz/glc+class.png",
    "6960~13860",
    ["중형 SUV"],
    ["10.7"]
  ),
  new CAR(
    "m6",
    ["c3"],
    "X5",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/bmw/x5.png",
    "10870~12420",
    ["준대형 SUV"],
    ["8~10.7"]
  ),
  new CAR(
    "m7",
    ["c4"],
    "Cruze",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/chevrolet/cruze.png",
    "1660~2478",
    ["준중형 세단"],
    ["13.5~16"]
  ),
  new CAR(
    "m8",
    ["c4"],
    "Malibu",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/chevrolet/malibu.png",
    "2364~3338",
    ["중형 세단"],
    ["10.5~14.2"]
  ),
  new CAR(
    "m9",
    ["c4"],
    "Orlando",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/chevrolet/Orlando.png",
    "2118~2916",
    ["중형 RV"],
    ["7.2~12.7"]
  ),
  new CAR(
    "m10",
    ["c4"],
    "Spark",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/chevrolet/spark.png",
    "977~1487",
    ["경형 해치백"],
    ["14.4~15"]
  ),
  new CAR(
    "m11",
    ["c4"],
    "Trax",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/chevrolet/trax.png",
    "1885~2333",
    ["소형 SUV"],
    ["11.8"]
  ),
  new CAR(
    "m12",
    ["c5"],
    "Explorer",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/ford/explorer.png",
    "6310~7160",
    ["준대형 SUV"],
    ["8.3~8.9"]
  ),
  new CAR(
    "m13",
    ["c6"],
    "EQ900",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/genesis/eq900.png",
    "7363~11584",
    ["대형 세단"],
    ["7.3~8.7"]
  ),
  new CAR(
    "m14",
    ["c6"],
    "G70",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/genesis/g70.png",
    "4035~4830",
    ["중형 세단"],
    ["8.9~14.9"]
  ),
  new CAR(
    "m15",
    ["c6"],
    "G80",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/genesis/g80.png",
    "5311~6251",
    ["준대형 세단"],
    ["8.4~14.6"]
  ),
  new CAR(
    "m16",
    ["c6"],
    "G90",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/genesis/g90.png",
    "8957~9307",
    ["대형 세단"],
    ["8.5~9.3"]
  ),
  new CAR(
    "m17",
    ["c7"],
    "Accent",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/accent.png",
    "1138~1913",
    ["소형 세단"],
    ["13.4~17.6"]
  ),
  new CAR(
    "m18",
    ["c7"],
    "Avante",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/avante.png",
    "1866~2806",
    ["준중형 세단"],
    ["10.5~15.4"]
  ),
  new CAR(
    "m19",
    ["c7"],
    "Grandeur",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/grandeur.png",
    "3392~4486",
    ["준대형 세단"],
    ["7.4~11.9"]
  ),
  new CAR(
    "m20",
    ["c7"],
    "i30",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/i30.png",
    "1865~2556",
    ["중형 해치백"],
    ["11.4~13"]
  ),
  new CAR(
    "m21",
    ["c7"],
    "ioniq",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/ioniq.png",
    "2242~2693",
    ["준중형 해치백"],
    ["22.4"]
  ),
  new CAR(
    "m22",
    ["c7"],
    "Kona",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/kona.png",
    "2144~3058",
    ["소형 SUV"],
    ["11.6~13.9"]
  ),
  new CAR(
    "m23",
    ["c7"],
    "Palisade",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/palisade.png",
    "3867~6028",
    ["준대형 SUV"],
    ["8.5~12.4"]
  ),
  new CAR(
    "m24",
    ["c7"],
    "Porter2",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/porter2.png",
    "1804~2366",
    ["중형 트럭"],
    ["8.6~9.5"]
  ),

  new CAR(
    "m25",
    ["c7"],
    "Santafe",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/santafe.png",
    "3252~4447",
    ["중형 SUV"],
    ["9.5~14.1"]
  ),
  new CAR(
    "m26",
    ["c7"],
    "Sonata",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/sonata.png",
    "2592~3633",
    ["중형 세단"],
    ["9.9~13.8"]
  ),
  new CAR(
    "m27",
    ["c7"],
    "Starex",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/starex.png",
    "2209~3280",
    ["대형 RV"],
    ["6.1~11"]
  ),
  new CAR(
    "m28",
    ["c7"],
    "Tucson",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/tucson.png",
    "2584~3801",
    ["준중형 SUV"],
    ["11~14.5"]
  ),
  new CAR(
    "m29",
    ["c7"],
    "Venue",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/hyundai/venue.png",
    "1689~2236",
    ["소형 SUV"],
    ["13.3~13.7"]
  ),
  new CAR(
    "m30",
    ["c8"],
    "Bongo3",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/bongo3.png",
    "1529~2219",
    ["중형 트럭"],
    ["6.5~9.9"]
  ),
  new CAR(
    "m31",
    ["c8"],
    "Carnival",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/carnival.png",
    "3180~4391",
    ["대형 RV"],
    ["8.9~13.1"]
  ),
  new CAR(
    "m32",
    ["c8"],
    "K3",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/k3.png",
    "1752~2449",
    ["준중형 세단"],
    ["14.1~15.2"]
  ),
  new CAR(
    "m33",
    ["c8"],
    "K5",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/k5.png",
    "2400~3284",
    ["중형 세단"],
    ["9.8~13.6"]
  ),
  new CAR(
    "m34",
    ["c8"],
    "K7",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/k7.png",
    "3137~3819",
    ["준대형 세단"],
    ["7.3~11.9"]
  ),
  new CAR(
    "m35",
    ["c8"],
    "K9",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/k9.png",
    "5694~7608",
    ["대형 세단"],
    ["8.1~9"]
  ),
  new CAR(
    "m36",
    ["c8"],
    "Mohave",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/mohave.png",
    "4958~5871",
    ["준대형 SUV"],
    ["9.3"]
  ),
  new CAR(
    "m37",
    ["c8"],
    "Morning",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/morning.png",
    "1175~1540",
    ["경형 해치백"],
    ["15.7"]
  ),
  new CAR(
    "m38",
    ["c8"],
    "Niro",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/niro.png",
    "2660~3306",
    ["소형 SUV"],
    ["19.1~20.8"]
  ),
  new CAR(
    "m39",
    ["c8"],
    "Ray",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/ray.png",
    "1340~1720",
    ["경형 RV"],
    ["12.7~13"]
  ),
  new CAR(
    "m40",
    ["c8"],
    "Sorento",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/sorento.png",
    "3002~4394",
    ["중형 SUV"],
    ["9.7~14.1"]
  ),
  new CAR(
    "m41",
    ["c8"],
    "Sportage",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/sportage.png",
    "2474~3818",
    ["준중형 SUV"],
    ["9~14.5"]
  ),
  new CAR(
    "m42",
    ["c8"],
    "Stinger",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/stinger.png",
    "3878~5006",
    ["중형 세단"],
    ["8.5~11.2"]
  ),
  new CAR(
    "m43",
    ["c8"],
    "Stonic",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/kia/stonic.png",
    "1625~2165",
    ["소형 SUV"],
    ["12.6~13.5"]
  ),
  new CAR(
    "m44",
    ["c9"],
    "Discovery",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/landrover/discovery.png",
    "8680~11340",
    ["준대형 SUV"],
    ["7.5~10.8"]
  ),
  new CAR(
    "m45",
    ["c9"],
    "Range Rover",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/landrover/rangerover.png",
    "20397~22437",
    ["대형 SUV"],
    ["6.8~10.1"]
  ),
  new CAR(
    "m46",
    ["c10"],
    "QM3",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/renaultsamsung/qm3.png",
    "2180~2523",
    ["소형 SUV"],
    ["17.3~17.4"]
  ),
  new CAR(
    "m47",
    ["c10"],
    "QM6",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/renaultsamsung/qm6.png",
    "2489~4075",
    ["중형 SUV"],
    ["8.6~12.5"]
  ),
  new CAR(
    "m48",
    ["c10"],
    "SM3",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/renaultsamsung/sm3.png",
    "1444~1763",
    ["준중형 세단"],
    ["13.8"]
  ),
  new CAR(
    "m49",
    ["c10"],
    "SM6",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/renaultsamsung/sm6.png",
    "2543~3530",
    ["중형 세단"],
    ["9.4~13.6"]
  ),
  new CAR(
    "m50",
    ["c11"],
    "Korando C",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/ssangyong/korando+c.png",
    "2202~2940",
    ["준중형 SUV"],
    ["11.8~14.3"]
  ),
  new CAR(
    "m51",
    ["c11"],
    "G4 Rexton",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/ssangyong/G4Rexton.png",
    "3448~4605",
    ["준대형 SUV"],
    ["10.1~10.5"]
  ),
  new CAR(
    "m52",
    ["c11"],
    "Rexton Sports",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/ssangyong/RextonSports.png",
    "2519~3940",
    ["준대형 트럭"],
    ["10.4~11.6"]
  ),
  new CAR(
    "m53",
    ["c11"],
    "Tivoli",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/ssangyong/Tivoli.png",
    "1659~2613",
    ["소형 SUV"],
    ["10.8~12.5"]
  ),
  new CAR(
    "m54",
    ["c12"],
    "Tiguan",
    "https://sticar-pjt.s3.ap-northeast-2.amazonaws.com/car/volkswagen/Tiguan.png",
    "4067~4705",
    ["준중형 SUV"],
    ["13.4~15.6"]
  ),
];
