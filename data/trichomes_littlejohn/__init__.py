from __future__ import annotations
from typing import TypeVar
import math
from pathlib import Path
from dataclasses import dataclass

from bioio import BioImage
import bioio_czi
from torch import Tensor
import torch

from ..volumetric import Metadata
from ..download_file_with_hash import ensure_cached_files_exist, cache_dir




_files = {
"1def1e0515bde6cfbcaff6d8a65460b31c8430d0d1ee183d29b2039e9a8d51a6":"trichomes_littlejohn/Nov_26th/11_19_01.czi",
"3af67cbb1493dfa15669785f9ea27a33761edab1666343e45371289218928b9b":"trichomes_littlejohn/Nov_26th/11_19_02.czi",
"007813390019926cf6e3109d407fafc9c3aa509de1a54ae8dd681d4b1df93a8c":"trichomes_littlejohn/Nov_26th/11_19_03.czi",
"8be3fc014adef542779830c0034d8692d7774e065c9aabc5b880d5e8790daea2":"trichomes_littlejohn/Nov_26th/11_19_04.czi",
"f08f0fa27ddb75b570872bb3cd34cd9873bcc4425402b0e2a8e8c2deba523cf7":"trichomes_littlejohn/Nov_26th/11_19_05.czi",
"f0e515e054e0027b45df139a9eb0b8545fe4c92603d62de2d80a601f9ac7b5c7":"trichomes_littlejohn/Nov_26th/11_19_06.czi",
"84489bd73ce134e7aef63f5a05ea8a6d5394ff16d3d956186dbad82c126d8945":"trichomes_littlejohn/Nov_26th/11_19_07.czi",
"fa808a50beb950e3bd74fe0c9619e1e48ce43b9e1e461bb5bcfd0106beb8b54b":"trichomes_littlejohn/Nov_26th/11_19_08.czi",
"221150026f0472a26c7481c570998aced51e83abfdaf02d1f1f272e2ea1fbc75":"trichomes_littlejohn/Nov_26th/11_19_09.czi",
"108f27f3db5576f22518cc7ca7dadb0c9b949280a4be387ddee613cc2b943391":"trichomes_littlejohn/Nov_26th/11_19_10.czi",
"5460e806d1f64ec1926fc047d81ba042cbd66c5f275072ddc6e9448eb9d8f481":"trichomes_littlejohn/Nov_26th/11_19_11.czi",
"e73cf694a963de2c053b65a38de10eb62077ed49ab1f244653b93c42d79cf5ec":"trichomes_littlejohn/Nov_26th/11_19_12.czi",
"052a626c9459d15d7176d8e15e6b5020d31227f945e6bab800b4d07cce8e75db":"trichomes_littlejohn/Nov_26th/11_19_13.czi",
"e5cb1a073e9f4a64399db9e0005eaae86284c9aaa6b3ea97d9983f4a70db43b9":"trichomes_littlejohn/Nov_26th/11_19_14.czi",
"4af13d0f63dfe389347adfcff712e4aadcc21c5aefbc4f2a7897c526536c1f63":"trichomes_littlejohn/Nov_26th/11_19_15.czi",
"eece87ab459d139486a6a357e5ca0a4fe30e8e5f3bf63e5508327c57d8e67524":"trichomes_littlejohn/Nov_26th/11_19_16.czi",
"dfa4e81d259b4d3a0af5d43efe35922e142cddb70c837ce4ed6c8822fa3a9285":"trichomes_littlejohn/Nov_26th/11_19_17.czi",
"1408fc52ec0b9494921b8e083e89ebe257ef12fd6c7ef1497de3e4a6fe9cb08b":"trichomes_littlejohn/Nov_26th/11_19_18.czi",
"f2976e14ac08d69e62051dbd056c5b64f5bb8493e06bba9190d16c9814fa7ff8":"trichomes_littlejohn/Nov_26th/11_19_19.czi",
"a9d3ca0ce345259cc3e6e5ed5a0687c4b04b7a8205598a64f5f1361bfca166c9":"trichomes_littlejohn/Nov_26th/11_19_20.czi",
"0d005830329179d13c28dad2048419a9c13ebbf03fc1bd9ef14c406f15aac3c3":"trichomes_littlejohn/Nov_26th/11_19_21.czi",
"fe75073c59d02815fee644a210ca824f8e872d1d2c38a952188e380aa42e7ba6":"trichomes_littlejohn/Nov_26th/11_19_22.czi",
"528bc0bcedf93b4d57597258d35f613a85747e136c68a840aa3ed2ca4ec9b9d3":"trichomes_littlejohn/Nov_26th/11_19_23.czi",
"5c671f52dc05f1438ca7c63b7fd1648ae9025106d54c51493dac3749bb3bf44a":"trichomes_littlejohn/Nov_26th/11_19_24.czi",
"dc1a4493d0581fa1ed5f10ce1006c59a7ae282c64f8f605b004e3db22d74495e":"trichomes_littlejohn/Nov_26th/11_19_25.czi",
"1c52544f8bf4508662ddd0f5394017956b984e15ebd320d2e389b74621c77677":"trichomes_littlejohn/Nov_26th/11_19_26.czi",
"ad73ee93e766b348763644629d06028cd1fa4a47a30cd686c280795540881b44":"trichomes_littlejohn/Nov_26th/11_19_27.czi",
"263b0f6d42402d718247bba3aef22acabc1ab9c2ebd2e19544f252b1a5c4bd86":"trichomes_littlejohn/Nov_26th/11_19_28.czi",
"8c3114116fa1f79106eb4925996c037040d91771aad6b33b63aa99a638610f3c":"trichomes_littlejohn/Nov_26th/11_19_29.czi",
"53b25215c1050fc231f0e5f12503700e5e40be883975b8faa0350b82d25db0eb":"trichomes_littlejohn/Nov_26th/11_19_30.czi",
"346bfe1ae77d13606f7fa39be1663f3896398c33f05caa0ebcd39c156c0e2e3f":"trichomes_littlejohn/Nov_26th/11_19_31.czi",
"67ccedf548d4c9eb603074a9dbaed9a3deb9ecad3ee8d2e132cf47ece0309a10":"trichomes_littlejohn/Nov_26th/11_19_32.czi",
"1fc371a64f7f8c56feb284c15a75cdf4e8bb757568cfa654fc6039f75d861436":"trichomes_littlejohn/Nov_26th/11_19_33.czi",
"e8e7a1fc80724b98833cb01bd9b1fb03ab9fe9c71662b50fb2755451f25e4fd7":"trichomes_littlejohn/Nov_26th/11_19_34.czi",
"ae4ec22042bb38ff7a1602ef0b193e74899f83e879afa5613d3fb5f7d84a5908":"trichomes_littlejohn/Nov_26th/11_19_35.czi",
"1f3db3419b9dbb2e00418e104a6a0dc4309b6c1e5adb98e46597134353b5a775":"trichomes_littlejohn/Nov_26th/11_19_36.czi",
"1c939c8417e8e4f2f803d8b21f3092849dc97eb8343811be735ea523cac482ec":"trichomes_littlejohn/Nov_26th/11_19_37.czi",
"02910309116cc1619b2740e14062c4b0ced7a165fe13f6fe7538d59639c76667":"trichomes_littlejohn/Nov_26th/11_19_38.czi",
"7f97bbcdeede6818d9bd304f622aca9d9f0d11372c9f0e88e963d427a5fdfc81":"trichomes_littlejohn/Nov_26th/11_19_39.czi",
"8a5a37e8a289417b0ab47559021dd196b678d8276a7a6d8953732401a695b28c":"trichomes_littlejohn/Nov_26th/11_19_40.czi",
"f04244a845c26e5de5a4a9fa2539a38376db881f052173e5233a9270dec96375":"trichomes_littlejohn/Nov_26th/11_19_41.czi",
"3012beef34c88f160fc08bfc84ba193720c85fa7cf26509fd86c2ad38d347a05":"trichomes_littlejohn/Nov_26th/11_19_42.czi",
"35a1a1c218cda14fbf5b83d62f63ccd104b41f03ad37beaae73c4616ccdf1d9d":"trichomes_littlejohn/Nov_26th/11_19_43.czi",
"bcc42be81eeb4ea84095551baa055e9d030fc353e56ebec12a6c3250aaebe9e9":"trichomes_littlejohn/Nov_26th/11_19_44.czi",
"e50c546222c730f4ab8a76e4e6968e617784cbc14696482aeb8c79d6dedb03cd":"trichomes_littlejohn/Nov_26th/11_19_45.czi",
"a22e4c50ef8fc3f1f74aa5c57b15be32fd0135e160d333fbc8dfd0236b9b5989":"trichomes_littlejohn/Nov_26th/11_19_46.czi",
"e88fb7c031d7779d837d25f89190a52c2b1adb3bf5a8e9c1b32deaa2245b78b1":"trichomes_littlejohn/Nov_26th/11_19_47.czi",
"803528137f6081337e95c9e136201bf906c98584b4378a6903f6306aeaaaa457":"trichomes_littlejohn/Nov_26th/11_19_48.czi",
"7bef9d67a3833abc0b5e20cb473c49d37a8aaa5dbf811fb57621f565f04a09f0":"trichomes_littlejohn/Nov_26th/11_19_49.czi",
"57860edabebc8420bf5e551550c7a46cdc6fcf2b8b622a44f99062cfc206d38a":"trichomes_littlejohn/Nov_26th/11_19_50.czi",
"85378efbfaa55f7ee42572130720474291c26d52c21d1974c59051dff38f708d":"trichomes_littlejohn/Nov_26th/11_19_51.czi",
"c02c1db60e741d7ad3fd1c5beaa0e07ae3f4d9be81f6c51ece30572feb36d324":"trichomes_littlejohn/Nov_26th/11_19_52.czi",
"a61e2f24de57c0b29142ea921ecb7f31ad11c68579bbc09112a4b311f2499b39":"trichomes_littlejohn/Nov_26th/11_19_53.czi",
"1bcdf1702f6bfb14f372dda0cc35c4372ab44f336a3a6146b35287d377ce3bc9":"trichomes_littlejohn/Nov_26th/11_19_54.czi",
"649826d199235697f02797ad686abf0984553a18495f739c3fe1e9e08e590148":"trichomes_littlejohn/Nov_26th/11_19_55.czi",
"1f9bf2ae4a4d7b0221746252cbaf10a9f8613e68412f3b3a782e29efc13adb9d":"trichomes_littlejohn/Nov_26th/11_19_56.czi",
"76c7f11ab74d5a74a81c92f55690b136548d919424633ebce92df4436c2f7a46":"trichomes_littlejohn/Nov_26th/11_19_57.czi",
"ef3e9e52680ee09f4a3dd27854d739affd19052764baed58a36cf291416be9ef":"trichomes_littlejohn/Nov_26th/11_19_58.czi",
"4f665e4f01927040a1dd632f44e38797c588c4ba03891323448075a16b5e50e4":"trichomes_littlejohn/Nov_26th/11_19_59.czi",
"9c307c80d7e72e1ebaecc0ea7f7d2bdd6cd14575b2f63dee8f5ef398007b0a95":"trichomes_littlejohn/Nov_26th/11_19_60.czi",
"b6ee51a0980029f9f22bd6ced9bab4dc42edf5e038432105697a1b0532f22d19":"trichomes_littlejohn/Nov_26th/11_19_61.czi",
"e04db1e230aec9c806d708b77176f71c03142bcd25f39b590046e3ace3273084":"trichomes_littlejohn/Nov_26th/11_19_62.czi",
"32190c1e521a6b1f2a64010b43530cdf6be77f49c6cd1d9afd315ef191fec930":"trichomes_littlejohn/Nov_26th/11_19_63.czi",
"e84d94c0abb51f8b21976d96f3e4dfeacda22749e0f06570cc5aa814770093e4":"trichomes_littlejohn/Oct_28th/10_28_01_128.czi",
"d501f383fa7bce842583bf2ba95cd0fae54303fdca9285622c54734efda83d2e":"trichomes_littlejohn/Oct_28th/10_28_02_128.czi",
"8570e726183a08a38b68963181fabeb63c047304f77a29f344c58151c66cbe8c":"trichomes_littlejohn/Oct_28th/10_28_03_128.czi",
"c3c44524d47c69d2bb3b554af6452daab22f7627eef9becea5e0b72b9a2ac90d":"trichomes_littlejohn/Oct_28th/10_28_04_128.czi",
"b1fd0202e5cf0b9880a28cc8e6b0621fff9905d2bd06d3436d24a846577c3814":"trichomes_littlejohn/Oct_28th/10_28_05_128.czi",
"a54634bc9ce008a8bf5e31ee8cf6114eb58c1a2505efba6c9e1371d242117ada":"trichomes_littlejohn/Oct_28th/10_28_06_128.czi",
"33499cd59430f0b393b7d9da12b04fb08803a52be6c75fd140341c82294d6c70":"trichomes_littlejohn/Oct_28th/10_28_07_128.czi",
"628d334adbde3938341adc65ccf4a87e846516c1548337f023d2431554548c31":"trichomes_littlejohn/Oct_28th/10_28_08_128.czi",
"1f3ee81a32cfdc9e19f628bcd42baa1a939750fabb4a893b35d9e18914e2ddf9":"trichomes_littlejohn/Oct_28th/10_28_09_128.czi",
"aba65a0aaa9e68062a6ec4fdee128e60bbe2698965e1fbacbb4bf12950e933eb":"trichomes_littlejohn/Oct_28th/10_28_10_128.czi",
"d9b13d0234e4a8947643eee0d38bcfa43fff5dbb036c715fb710515a6cb75dfa":"trichomes_littlejohn/Oct_28th/10_28_11_128.czi",
"087feac4da0ff35ad42736fa8c2598eb9934923a7a39adefc917fa719c90be01":"trichomes_littlejohn/Oct_28th/10_28_12_128.czi",
"9d37f05ef511a370e3998286b60a345727b533a00202dc7b7a2516d70cf8d2b6":"trichomes_littlejohn/Oct_28th/10_28_13_128.czi",
"bca013d5acb3ba4c61315f7226da7925daf8a4593de0ba54ab4226000d1e413f":"trichomes_littlejohn/Oct_28th/10_28_14_128.czi",
"1d2719159b781d11455b83477ceb99fb2246d3b0a697acdb87040a6267d64c79":"trichomes_littlejohn/Oct_28th/10_28_15_128.czi",
"24c0b7006e4b19eaccd808cb2098699a6e7aabedf805f04b4deb6c7eb7fffeb4":"trichomes_littlejohn/Oct_28th/10_28_16_128.czi",
"bb91b14dfc3fd9f48353f50fb78ad67a1859d909d65b8999d664e078b72acff4":"trichomes_littlejohn/Oct_28th/10_28_17_128.czi",
"ed4a271ea5788da90a4bdef67cfb2d5466ff5bb8d8c1d99fd0251ab5f5e63bd8":"trichomes_littlejohn/Oct_28th/10_28_18_128.czi",
"22ebb0395fe25263a696dc2c794bdeda84ac07154a86d48a6fb30954a2f2623b":"trichomes_littlejohn/Oct_28th/10_28_19_128.czi",
"7f1a5e0a408f72528c5db2bbfa7a43c174175bf2b39ea67fbeffd7c8adc1e76a":"trichomes_littlejohn/Oct_28th/10_28_21_128.czi",
"0c83157cad7243ed5a63d840975a8dbe87084ea2ba3ffd4c0c1bcf67a9f39938":"trichomes_littlejohn/Oct_28th/10_28_22_128.czi",
"e4695448bf506d092562472a9e910020677228e27ec8bbb4c96108397d7c70ba":"trichomes_littlejohn/Oct_28th/10_28_23_128.czi",
"e9f5c0b2854bb74e6be4776f24a820fcc43e7b7a06f29b7af94a88d7fa09e437":"trichomes_littlejohn/Oct_28th/10_28_24_128.czi",
"8e4e00b06e0ff70944ca38ce1bd3586d9b26078452fb102bf7026c264f5940f8":"trichomes_littlejohn/Oct_28th/10_28_25_128.czi",
"f488511c2ea80ebe7e828e76674bb1fd041153889a0754b64582e4dfc4084de2":"trichomes_littlejohn/Oct_28th/10_28_26_128.czi",
"afcfdc393d577993f0aa7be22b56c6e1fe063f2fe90dce1e455a355e29ab86d7":"trichomes_littlejohn/Oct_28th/10_28_27_128.czi",
"cce262b8bd7d63247f4a3188208ecf2fac0d38a619bde2df4fa9bfb3e3a65879":"trichomes_littlejohn/Oct_28th/10_28_28_128.czi",
"1168f10e0b7b2645fa11587e92d4c70fc0f19d512ff689fdd35733d87a5af7a4":"trichomes_littlejohn/Oct_28th/10_28_29_128.czi",
"7c0cf85cef94ad54900118177e6831abfec8aab1e204015d583d98d4ca3ae5f3":"trichomes_littlejohn/Oct_28th/10_28_30_128.czi",
"44cf6bec302972b930144967bc4d39804b906d2f9a6e1283adb83e0aef8bbdaf":"trichomes_littlejohn/Oct_28th/10_28_31_128.czi",
"6659cb46426d81f3b9046eebc36d182517f0e7dd6164234739199e597e875400":"trichomes_littlejohn/Oct_28th/10_28_32_128.czi",
"91f37ec0e69f52b7ed9f63173c9573eebb367dd2acb6e0d8d244ffaf9b252185":"trichomes_littlejohn/Oct_28th/10_28_33_128.czi",
"e11c95adaeca2e7727537ae581f80a004d9954c6ff76cdb8a270017a5efb35ba":"trichomes_littlejohn/Oct_28th/10_28_34_128.czi",
"d7a61b6e94eabaf3f3199264a8e3e5606458435d43da3f0f13106c5cc0306e56":"trichomes_littlejohn/Oct_28th/10_28_35_128.czi",
"77553c9d5136ac2065b3374652ab352bc1bc4a96b2935c0101fe5d111d20a785":"trichomes_littlejohn/Oct_28th/10_28_36_128.czi",
"256fa10d6616bf29d7dfa14f7882fe48fea818cc52574c9220f6c9f6f68553c0":"trichomes_littlejohn/Oct_28th/10_28_37_128.czi",
"924a8aadc983914149a53df808d0d64648af51250990972db026221802b73d09":"trichomes_littlejohn/Oct_28th/10_28_38_128.czi",
"89071134534e828b395f8ebad61e9de2fab55e3f2dfcb8761272b5b3bd444518":"trichomes_littlejohn/Oct_28th/10_28_39_128.czi",
"2456f8d4c51033b47995c1d7d5b7129367988708e560f52fa5ca688f6dc649f3":"trichomes_littlejohn/Oct_28th/10_28_40_128.czi",
"4fc331742a73e37e69d2a54ae2c54612ed219cf7ba4e195e76a4fa25ff36e713":"trichomes_littlejohn/Oct_28th/10_28_41_128.czi",
"3c0551f1aecbb38a08111309817eee7a4befb96ad0f9d2464b56dcdea7cc7931":"trichomes_littlejohn/OneDrive_1_10-17-2024/01.czi",
"7ad80381a942258ff4a256d4ce5f57c3cb6e8fd5d2a4da9c2b6f88d9bd0c4bfd":"trichomes_littlejohn/OneDrive_1_10-17-2024/02.czi",
"cb297c9cc54cc82af152fd6d45119b6debe992fae578b318abbbdd813fb5ae28":"trichomes_littlejohn/OneDrive_1_10-17-2024/03.czi",
"f5cbcef3cb242908ac2d4cde34789675a1be0a6837889a7a8e82a7a358ae4029":"trichomes_littlejohn/OneDrive_1_10-17-2024/04.czi",
"e376b134ec06d0aceecd0e5dc8f0f157073458474ab272bf1ad56cf94374e18b":"trichomes_littlejohn/OneDrive_1_10-17-2024/05.czi",
"579fb65e34762b1b70f53d5dd6c8e71d6e77bd408af67cdcf1c1a93e09d991be":"trichomes_littlejohn/OneDrive_1_10-17-2024/06.czi",
"ce3e23bc3bf734425dd437699fefcb38ef9ace2d64e96bbcac473fa99b5ebf19":"trichomes_littlejohn/OneDrive_1_10-17-2024/07.czi",
"da74204d1a0f08242a1f0025ce2bec71c67dbec2550ad8cc6dfadd45390c86bb":"trichomes_littlejohn/OneDrive_1_10-17-2024/08.czi",
"63119ea98c8411d789270e6565e32a3237a226ba76508b212c2cf481f0273794":"trichomes_littlejohn/OneDrive_1_10-17-2024/09.czi",
"c4fd6b7aac4072d522fd6baf77716bf844b038adf01b5d93543df851e78061db":"trichomes_littlejohn/OneDrive_1_10-17-2024/10.czi",
"750aef393d3f2df1a33b609f7340b71008fdd976c3a1dcc18ecb65ae63c395f4":"trichomes_littlejohn/OneDrive_1_10-17-2024/11.czi",
"b1c96d0cc52aee9b96778c72c6ee614160ee0c1d430edc3b256009383debe66c":"trichomes_littlejohn/OneDrive_1_10-17-2024/12.czi",
}


_FILES_10_17_2024 = [
    "OneDrive_1_10-17-2024/01.czi",
    "OneDrive_1_10-17-2024/02.czi",
    "OneDrive_1_10-17-2024/03.czi",
    "OneDrive_1_10-17-2024/04.czi",
    "OneDrive_1_10-17-2024/05.czi",
    "OneDrive_1_10-17-2024/06.czi",
    "OneDrive_1_10-17-2024/07.czi",
    "OneDrive_1_10-17-2024/08.czi",
    "OneDrive_1_10-17-2024/09.czi",
    "OneDrive_1_10-17-2024/10.czi",
    "OneDrive_1_10-17-2024/11.czi",
    "OneDrive_1_10-17-2024/12.czi",
]

_labels_10_17_2024 = Path(__file__).parent/'OneDrive_1_10-17-2024'/'labels.zip'

_FILES_Oct_28 = [
    "Oct_28th/10_28_01_128.czi",
    "Oct_28th/10_28_02_128.czi",
    "Oct_28th/10_28_03_128.czi",
    "Oct_28th/10_28_04_128.czi",
    "Oct_28th/10_28_05_128.czi",
    "Oct_28th/10_28_06_128.czi",
    "Oct_28th/10_28_07_128.czi",
    "Oct_28th/10_28_08_128.czi",
    "Oct_28th/10_28_09_128.czi",
    "Oct_28th/10_28_10_128.czi",
    "Oct_28th/10_28_11_128.czi",
    "Oct_28th/10_28_12_128.czi",
    "Oct_28th/10_28_13_128.czi",
    "Oct_28th/10_28_14_128.czi",
    "Oct_28th/10_28_15_128.czi",
    "Oct_28th/10_28_16_128.czi",
    "Oct_28th/10_28_17_128.czi",
    "Oct_28th/10_28_18_128.czi",
    "Oct_28th/10_28_19_128.czi",
    "Oct_28th/10_28_21_128.czi",
    "Oct_28th/10_28_22_128.czi",
    "Oct_28th/10_28_23_128.czi",
    "Oct_28th/10_28_24_128.czi",
    "Oct_28th/10_28_25_128.czi",
    "Oct_28th/10_28_26_128.czi",
    "Oct_28th/10_28_27_128.czi",
    "Oct_28th/10_28_28_128.czi",
    "Oct_28th/10_28_29_128.czi",
    "Oct_28th/10_28_30_128.czi",
    "Oct_28th/10_28_31_128.czi",
    "Oct_28th/10_28_32_128.czi",
    "Oct_28th/10_28_33_128.czi",
    "Oct_28th/10_28_34_128.czi",
    "Oct_28th/10_28_35_128.czi",
    "Oct_28th/10_28_36_128.czi",
    "Oct_28th/10_28_37_128.czi",
    "Oct_28th/10_28_38_128.czi",
    "Oct_28th/10_28_39_128.czi",
    "Oct_28th/10_28_40_128.czi",
    "Oct_28th/10_28_41_128.czi",
]
_labels_Oct_28 = Path(__file__).parent/'Oct_28th'/'labels.zip'

_FILES_Nov_26 = [
    "Nov_26th/11_19_01.czi",
    "Nov_26th/11_19_02.czi",
    "Nov_26th/11_19_03.czi",
    "Nov_26th/11_19_04.czi",
    "Nov_26th/11_19_05.czi",
    "Nov_26th/11_19_06.czi",
    "Nov_26th/11_19_07.czi",
    "Nov_26th/11_19_08.czi",
    "Nov_26th/11_19_09.czi",
    "Nov_26th/11_19_10.czi",
    "Nov_26th/11_19_11.czi",
    "Nov_26th/11_19_12.czi",
    "Nov_26th/11_19_13.czi",
    "Nov_26th/11_19_14.czi",
    "Nov_26th/11_19_15.czi",
    "Nov_26th/11_19_16.czi",
    "Nov_26th/11_19_17.czi",
    "Nov_26th/11_19_18.czi",
    "Nov_26th/11_19_19.czi",
    "Nov_26th/11_19_20.czi",
    "Nov_26th/11_19_21.czi",
    "Nov_26th/11_19_22.czi",
    "Nov_26th/11_19_23.czi",
    "Nov_26th/11_19_24.czi",
    "Nov_26th/11_19_25.czi",
    "Nov_26th/11_19_26.czi",
    "Nov_26th/11_19_27.czi",
    "Nov_26th/11_19_28.czi",
    "Nov_26th/11_19_29.czi",
    "Nov_26th/11_19_30.czi",
    "Nov_26th/11_19_31.czi",
    "Nov_26th/11_19_32.czi",
    "Nov_26th/11_19_33.czi",
    "Nov_26th/11_19_34.czi",
    "Nov_26th/11_19_35.czi",
    "Nov_26th/11_19_36.czi",
    "Nov_26th/11_19_37.czi",
    "Nov_26th/11_19_38.czi",
    "Nov_26th/11_19_39.czi",
    "Nov_26th/11_19_40.czi",
    "Nov_26th/11_19_41.czi",
    "Nov_26th/11_19_42.czi",
    "Nov_26th/11_19_43.czi",
    "Nov_26th/11_19_44.czi",
    "Nov_26th/11_19_45.czi",
    "Nov_26th/11_19_46.czi",
    "Nov_26th/11_19_47.czi",
    "Nov_26th/11_19_48.czi",
    "Nov_26th/11_19_49.czi",
    "Nov_26th/11_19_50.czi",
    "Nov_26th/11_19_51.czi",
    "Nov_26th/11_19_52.czi",
    "Nov_26th/11_19_53.czi",
    "Nov_26th/11_19_54.czi",
    "Nov_26th/11_19_55.czi",
    "Nov_26th/11_19_56.czi",
    "Nov_26th/11_19_57.czi",
    "Nov_26th/11_19_58.czi",
    "Nov_26th/11_19_59.czi",
    # "Nov_26th/11_19_60.czi", # This one has a weird resolution, 10% off 
    "Nov_26th/11_19_61.czi",
    "Nov_26th/11_19_62.czi",
    "Nov_26th/11_19_63.czi",
]
_labels_Nov_26 = Path(__file__).parent/'Nov_26th'/'labels.zip'


@dataclass
class Label:
    """Forgot to put _, too late to make intenal now lol"""
    angle: float
    point_1: Tensor|None = None
    point_2: Tensor|None = None


def _R_Y(angle: float, device:torch.device|None=None)->Tensor:
    c = math.cos(angle)
    s = math.sin(angle)

    R = torch.tensor([
        [  c,  0,  s],
        [  0,  1,  0],
        [ -s,  0,  c],
    ], device=device)

    return R


# Note vol coords * _vol_nm_scale_xyz should give the same results as
# (torch.arange(shape[2])-(shape[2]-1)/2) * metadata.xy_nm_pix
def _vol_coords_xyz(shape: torch.Size, d:torch.device|None=None)->Tensor:
    # Coordinate lookup is always done in the range [-1,1], as per OpenGL
    xs = torch.arange(shape[2], device=d).unsqueeze(0).unsqueeze(0).expand(shape) / (shape[2]-1) * 2 - 1
    ys = torch.arange(shape[1], device=d).unsqueeze(0).unsqueeze(2).expand(shape) / (shape[1]-1) * 2 - 1
    zs = torch.arange(shape[0], device=d).unsqueeze(1).unsqueeze(2).expand(shape) / (shape[0]-1) * 2 - 1
    return torch.stack([xs, ys, zs], 3)


def _vol_nm_scale_xyz(shape: torch.Size, metadata: Metadata)->torch.Tensor:
    # xscale goes from -1 to 1
    # -1 corresponts to -x_size/2 * nm_per_pix
    # 1 corresponds to  x_size/2 * nm_per_pix

    # Scale volume coords to nanometers
    x_scale = (shape[2]-1)/2 * metadata.xy_nm_pix
    y_scale = (shape[1]-1)/2 * metadata.xy_nm_pix
    z_scale = (shape[0]-1)/2 * metadata.z_nm_pix
    return torch.tensor([x_scale, y_scale, z_scale])





def _cut_volume(volume: Tensor, metadata: Metadata, angle: float, point_1: Tensor, point_2: Tensor)->Tensor:
    d = volume.device
    
    R = _R_Y(angle, d).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(*volume.shape, 3, 3)
    scale = _vol_nm_scale_xyz(volume.shape, metadata).reshape(1,1,1,3).expand(*volume.shape, 3)
    xyz_nm = _vol_coords_xyz(volume.shape, d) * scale
    rotated_coords_nm = (R @ xyz_nm.unsqueeze(-1)).squeeze(-1) 

    point_1_nm = (point_1 - (torch.tensor([volume.shape[2], volume.shape[1]])/2-1)) * metadata.xy_nm_pix
    point_2_nm = (point_2 - (torch.tensor([volume.shape[2], volume.shape[1]])/2-1)) * metadata.xy_nm_pix

    vec2 = point_1_nm - point_2_nm

    vec3 = torch.cat([vec2, torch.zeros(1)]) #This vector is on the plane
    other_vec3 = torch.tensor([0,0,1.]).to(vec3) # The plane goes in Z, in the rotated frame
    
    normal = torch.linalg.cross(vec3, other_vec3) # pylint: disable=not-callable

    plane_point = torch.cat([point_1_nm, torch.zeros(1)]).reshape(1,1,1,3).expand_as(rotated_coords_nm)

    cut  =  (normal.reshape(1,1,1,1,3).expand(*volume.shape, 1, 3) @ (rotated_coords_nm - plane_point).unsqueeze(-1)) < 0
    cut = cut.squeeze(-1).squeeze(-1)

    cut_volume = volume.clone()
    cut_volume[cut] = 0
    return cut_volume



T = TypeVar('T')

def _not_none(a: T|None)->T:
    if a is None:
        raise RuntimeError('Expected value, got None')
    return a


def _approx_eq(a: float, b: float, epsilon: float)->bool:
    return abs(a-b)/(a+b) < epsilon

def _load_file(filename: Path)->tuple[Tensor, Metadata]:
    img = BioImage(filename, reader=bioio_czi.Reader)

    if img.physical_pixel_sizes.Y is None or img.physical_pixel_sizes.X is None:
        raise RuntimeError(f'Resolution missing in {filename}')

    if not _approx_eq(img.physical_pixel_sizes.Y, img.physical_pixel_sizes.X, 1e-8):
        raise RuntimeError(f'Resolution mismatch in {filename}')

    fwhm_ratio = 1/3. # effective PSF of the resulting data cube
    rx = _not_none(img.physical_pixel_sizes.X)
    rz = _not_none(img.physical_pixel_sizes.Z)
    metadata = Metadata(
        xy_nm_pix = rx, 
        z_nm_pix = rz,
        xy_fwhm_nm = rx * fwhm_ratio,
        z_fwhm_nm = rz * fwhm_ratio,
    )

    return torch.tensor(img.data[0,0,...]), metadata

def _pad(stack: Tensor, xy_size: int, z_size: int)->Tensor:

    
    padded = torch.zeros(z_size, xy_size, xy_size)
    off = (torch.tensor(padded.shape)-torch.tensor(stack.shape))//2
    slices = [ slice(i, i+j) for i,j in zip(off, stack.shape)]
    #print(slices)
    padded[*slices] = stack
    return padded

def _concat(loaded: list[tuple[Tensor, Metadata]], xy_size: int, z_size: int)->tuple[Tensor, Metadata]:
     
    xy_nm_pix = loaded[0][1].xy_nm_pix
    z_nm_pix = loaded[0][1].z_nm_pix
    good = [ _approx_eq(xy_nm_pix, i[1].xy_nm_pix, 5e-3) and _approx_eq(z_nm_pix, i[1].z_nm_pix, 1e-3) for i in loaded]

    if not all(good):
        for (_,m),g in zip(loaded, good):
            print(m, "" if g else "*"*10)
        raise RuntimeError('Metadata mismatch')
    
    return torch.stack([_pad(i[0], xy_size, z_size) for i in loaded], 0), loaded[0][1]

def _half_xy(stacks: Tensor, metadata: Metadata)->tuple[Tensor, Metadata]:
    scaled_stacks = torch.nn.functional.avg_pool2d(stacks, 2) # pylint: disable=not-callable

    scaled_metadata = Metadata(
        xy_nm_pix = metadata.xy_nm_pix * 2,
        z_nm_pix = metadata.z_nm_pix,
        xy_fwhm_nm = metadata.xy_fwhm_nm * 2,
        z_fwhm_nm = metadata.z_fwhm_nm
    )
    return scaled_stacks, scaled_metadata


def _load_local_list(files: list[str])->list[tuple[Tensor, Metadata]]:
    return [ _load_file(cache_dir/"trichomes_littlejohn"/i) for i in files]

_FINAL_SIZE = (178,60)

def _load_and_reshape(files: list[str])->tuple[Tensor, Metadata]:
    loaded = _load_local_list(files)
    return _half_xy(*_concat(loaded, *_FINAL_SIZE))


def _process_labels(dataset: torch.Tensor, metadata: Metadata, labels: list[Label|None])->Tensor:
    cuts = dataset.clone()

    for label, i in zip(labels, range(len(dataset))):
        if label is not None:
            angle = label.angle
            point_1 = label.point_1
            point_2 = label.point_2
            if angle is not None and point_1 is not None and point_2 is not None:
                cuts[i] = _cut_volume(dataset[i], metadata, angle*torch.pi/180, point_1, point_2)
    
    return cuts

def _load_labelled_data(files: list[str], label_file: Path)->tuple[Tensor, Metadata]:
    dataset, metadata = _load_and_reshape(files)
    labels = torch.load(label_file)
    return _process_labels(dataset, metadata, labels), metadata


def load_all()->tuple[Tensor, Metadata]:
    '''Load all the data. Duh'''
    
    ensure_cached_files_exist(_files)

    all_data_files = [
        (_FILES_10_17_2024, _labels_10_17_2024),
        (_FILES_Oct_28, _labels_Oct_28),
        (_FILES_Nov_26, _labels_Nov_26)
    ]

    all_data_and_metadata = [ _load_labelled_data(i, j) for i, j in all_data_files]

    all_data = [i[0] for i in all_data_and_metadata]

    if not all(i[1] == all_data_and_metadata[0][1] for i in all_data_and_metadata):
        raise RuntimeError('Metadata mismatch')

    return torch.cat(all_data, 0), all_data_and_metadata[0][1]
