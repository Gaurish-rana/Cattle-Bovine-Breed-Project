export interface BreedCharacteristics {
  origin: string;
  primary_use: string;
  average_weight: string;
  distinctive_features: string;
  species?: string;
  accession_number?: string;
}

export interface BreedData {
  id: string;
  name: string;
  image: string;
  species: 'Cattle' | 'Buffalo';
  accession_number: string;
  characteristics: BreedCharacteristics;
}

export interface PredictionResult {
  rank?: number;
  breed_id: string;
  breed_name: string;
  species: 'Cattle' | 'Buffalo';
  confidence: number;
  confidence_percentage?: string;
  characteristics: BreedCharacteristics;
}

export interface RankedProbability {
  rank: number;
  breed_id: string;
  breed_name: string;
  species: string;
  confidence: number;
  confidence_percentage: string;
}

export interface APIResponse {
  success: boolean;
  model_loaded?: boolean;
  architecture?: string;
  device?: string;
  inference_time_ms?: number;
  message?: string;
  predictions: PredictionResult[];
  all_probabilities?: RankedProbability[];
  disclaimer?: string;
  error?: string;
}

// 17 Indian indigenous cattle and buffalo breeds verified by ICAR-NBAGR
export const mockBreedData: BreedData[] = [
  // CATTLE (8 Breeds)
  {
    id: 'amritmahal',
    name: 'Amritmahal',
    species: 'Cattle',
    accession_number: 'INDIA_CATTLE_0800_AMRITMAHAL_03001',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Hassan, Chikmagalur & Tumkur, Karnataka, India',
      primary_use: 'Draught work (renowned cavalry draught breed)',
      average_weight: '380 - 450 kg (females) | 450 - 520 kg (males)',
      distinctive_features: 'Grey to dark iron-grey coat; long, sharp, upward-curving horns tapering to sharp points; compact muscular body frame.'
    }
  },
  {
    id: 'gir',
    name: 'Gir',
    species: 'Cattle',
    accession_number: 'INDIA_CATTLE_0400_GIR_03007',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Saurashtra / Gir Hills, Gujarat, India',
      primary_use: 'Dairy (High A2 milk producer)',
      average_weight: '380 - 450 kg (females) | 500 - 650 kg (males)',
      distinctive_features: 'Convex domed forehead shield; long pendulous folded ears resembling curled leaves; mottled red and white or pure red coat.'
    }
  },
  {
    id: 'hallikar',
    name: 'Hallikar',
    species: 'Cattle',
    accession_number: 'INDIA_CATTLE_0800_HALLIKAR_03008',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Mysore, Mandya & Hassan, Karnataka, India',
      primary_use: 'Draught work (fast trotting and road haulage)',
      average_weight: '340 - 420 kg (females) | 420 - 500 kg (males)',
      distinctive_features: 'Dark grey to grey coat with lighter belly; long, erect, backward-swept horns curving inward; agile high-speed gait.'
    }
  },
  {
    id: 'kankrej',
    name: 'Kankrej',
    species: 'Cattle',
    accession_number: 'INDIA_CATTLE_0417_KANKREJ_03011',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Rann of Kutch & Banaskantha, Gujarat, India',
      primary_use: 'Dual purpose (Dairy and powerful draught)',
      average_weight: '420 - 500 kg (females) | 550 - 700 kg (males)',
      distinctive_features: 'Magnificent lyre-shaped horns; silver-grey to iron-grey coloration; characteristic high-stepping majestic gait (Sawai Chaal).'
    }
  },
  {
    id: 'krishnavalley',
    name: 'Krishna Valley',
    species: 'Cattle',
    accession_number: 'INDIA_CATTLE_0800_KRISHNAVALLEY_03015',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Krishna River Basin, Karnataka & Maharashtra, India',
      primary_use: 'Heavy draught and agricultural tillage',
      average_weight: '450 - 550 kg (females) | 600 - 750 kg (males)',
      distinctive_features: 'Massive heavy frame with deep chest; grey-white body with darker extremities in bulls; short curved horns.'
    }
  },
  {
    id: 'nimari',
    name: 'Nimari',
    species: 'Cattle',
    accession_number: 'INDIA_CATTLE_1000_NIMARI_03019',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Narmada Valley (Khargone & Khandwa), Madhya Pradesh, India',
      primary_use: 'Draught and milk (Gir x Khillari cross derivative)',
      average_weight: '350 - 420 kg (females) | 450 - 550 kg (males)',
      distinctive_features: 'Copper-red coat with prominent white splashes; moderately convex forehead; hardy constitution suited to hot dry climates.'
    }
  },
  {
    id: 'ongole',
    name: 'Ongole',
    species: 'Cattle',
    accession_number: 'INDIA_CATTLE_0100_ONGOLE_03020',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Prakasam & Guntur, Andhra Pradesh, India',
      primary_use: 'Dual purpose (Heavy draught and beef progenitor worldwide)',
      average_weight: '450 - 550 kg (females) | 600 - 800 kg (males)',
      distinctive_features: 'Glossy porcelain-white coat; majestic muscular hump; dark muzzle and eye rims; large pendulous dewlap.'
    }
  },
  {
    id: 'vechur',
    name: 'Vechur',
    species: 'Cattle',
    accession_number: 'INDIA_CATTLE_0900_VECHUR_03030',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Kottayam & Alleppey districts, Kerala, India',
      primary_use: 'Dairy (Medicinal A2 milk with tiny fat globules)',
      average_weight: '130 - 180 kg (females) | 160 - 220 kg (males)',
      distinctive_features: 'Guinness World Record smallest cattle breed (height < 90 cm); light red, brown, or white coat; tiny forward-pointing horns.'
    }
  },

  // BUFFALO (9 Breeds)
  {
    id: 'banni',
    name: 'Banni',
    species: 'Buffalo',
    accession_number: 'INDIA_BUFFALO_0400_BANNI_01011',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Kutch & Banni Grasslands, Gujarat, India',
      primary_use: 'High-yield milk production (night grazing adapted)',
      average_weight: '450 - 550 kg (females) | 550 - 680 kg (males)',
      distinctive_features: 'Solid black or copper-brown coat; tightly coiled spiral vertical horns; exceptional capacity for nocturnal foraging.'
    }
  },
  {
    id: 'bargur',
    name: 'Bargur',
    species: 'Buffalo',
    accession_number: 'INDIA_BUFFALO_1800_BARGUR_01015',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Bargur Forest Hills (Erode), Tamil Nadu, India',
      primary_use: 'Draught and milk in hilly forest terrains',
      average_weight: '350 - 450 kg (females) | 400 - 500 kg (males)',
      distinctive_features: 'Brownish-black coat with occasional white markings; semicircular horns pointing backward and upward; agile hill climber.'
    }
  },
  {
    id: 'bhadawari',
    name: 'Bhadawari',
    species: 'Buffalo',
    accession_number: 'INDIA_BUFFALO_2010_BHADAWARI_01003',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Ravines of Yamuna & Chambal (Agra & Etawah), Uttar Pradesh, India',
      primary_use: 'High-fat dairy (world-leading milk fat content up to 13%)',
      average_weight: '350 - 425 kg (females) | 450 - 550 kg (males)',
      distinctive_features: 'Copper / wheat-straw colored coat (light brown); two white chevrons (\'Janeu\') on the lower neck; curled horns.'
    }
  },
  {
    id: 'jaffarabadi',
    name: 'Jaffarabadi',
    species: 'Buffalo',
    accession_number: 'INDIA_BUFFALO_0400_JAFFARABADI_01006',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Gir Forest & Coastal Saurashtra, Gujarat, India',
      primary_use: 'High-fat dairy and heavy draught',
      average_weight: '600 - 750 kg (females) | 800 - 1000 kg (males)',
      distinctive_features: 'Largest and heaviest Indian river buffalo; prominent protruding frontal bones; heavy drooping horns that curl upward at tips.'
    }
  },
  {
    id: 'mehsana',
    name: 'Mehsana',
    species: 'Buffalo',
    accession_number: 'INDIA_BUFFALO_0400_MEHSANA_01004',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Mehsana, Banaskantha & Sabarkantha, Gujarat, India',
      primary_use: 'Dairy (High persistent lactation; Murrah x Surti cross)',
      average_weight: '450 - 550 kg (females) | 550 - 680 kg (males)',
      distinctive_features: 'Jet black to brown coat; sickle-curved horns less tightly coiled than Murrah; elongated head and persistent lactation cycle.'
    }
  },
  {
    id: 'murrah',
    name: 'Murrah',
    species: 'Buffalo',
    accession_number: 'INDIA_BUFFALO_0500_MURRAH_01001',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Rohtak, Jind & Hisar, Haryana, India',
      primary_use: 'Premier high-yield dairy breed worldwide',
      average_weight: '500 - 650 kg (females) | 650 - 850 kg (males)',
      distinctive_features: 'Jet-black coat; tightly curled spiral horns; short, broad forehead; wedge-shaped dairy conformation with prominent udder.'
    }
  },
  {
    id: 'nagpuri',
    name: 'Nagpuri',
    species: 'Buffalo',
    accession_number: 'INDIA_BUFFALO_1100_NAGPURI_01007',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Vidarbha Region (Nagpur & Wardha), Maharashtra, India',
      primary_use: 'Dual purpose (Draught and milk in hot arid conditions)',
      average_weight: '400 - 500 kg (females) | 500 - 600 kg (males)',
      distinctive_features: 'Long, flat, sword-shaped horns sweeping backwards to the shoulders; black coat; heat-tolerant Ellichpuri type.'
    }
  },
  {
    id: 'niliravi',
    name: 'Nili Ravi',
    species: 'Buffalo',
    accession_number: 'INDIA_BUFFALO_1600_NILIRAVI_01002',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Sutlej & Ravi River Valleys (Ferozepur, Punjab), India / Pakistan',
      primary_use: 'High-yield dairy milk production',
      average_weight: '500 - 650 kg (females) | 650 - 800 kg (males)',
      distinctive_features: '\'Panj Kalyan\' (five white markings: forehead blaze, muzzle, all 4 legs, and tail switch); walled light-colored eyes; tightly coiled horns.'
    }
  },
  {
    id: 'toda',
    name: 'Toda',
    species: 'Buffalo',
    accession_number: 'INDIA_BUFFALO_0018_TODA_01010',
    image: '/placeholder.svg',
    characteristics: {
      origin: 'Nilgiri Hills plateau, Tamil Nadu, India',
      primary_use: 'Milk and ritual sacred dairy herds of Toda pastoralists',
      average_weight: '420 - 500 kg (females) | 480 - 580 kg (males)',
      distinctive_features: 'Wide crescent-shaped horns spreading outward and curving inward; thick silver-grey/fawn hair coat with dark dorsal stripe.'
    }
  }
];

export const API_BASE_URL = 'http://localhost:5000';

export const getBreedById = (id: string): BreedData | undefined => {
  return mockBreedData.find(breed => breed.id.toLowerCase() === id.toLowerCase());
};

export const ALL_BREEDS = [
  'amritmahal', 'banni', 'bargur', 'bhadawari', 'gir', 'hallikar',
  'jaffarabadi', 'kankrej', 'krishnavalley', 'mehsana', 'murrah',
  'nagpuri', 'niliravi', 'nimari', 'ongole', 'toda', 'vechur'
];
