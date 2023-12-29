from loguru import logger


class AuditoryConstants():

    @staticmethod
    def AGE_BOXING(age: int):
        if 15 <= age <= 24:
            return "20"
        elif 25 <= age <= 34:
            return "30"
        elif 35 <= age <= 44:
            return "40"
        elif 45 <= age <= 54:
            return "50"
        elif 55 <= age <= 64:
            return "60"
        else:
            logger.warning(f"age={age} in not valid in table!")
            return "60"

    STANDARD_PTA_DICT = {
        "Male": {
            "20": {
                "500Hz": {"10pr": 3, "50pr": 6, "90pr": 10, },
                "1000Hz": {"10pr": 1, "50pr": 6, "90pr": 12, },
                "2000Hz": {"10pr": -2, "50pr": 4, "90pr": 11, },
                "3000Hz": {"10pr": 1, "50pr": 5, "90pr": 12, },
                "4000Hz": {"10pr": -3, "50pr": 6, "90pr": 12, },
                "6000Hz": {"10pr": 3, "50pr": 12, "90pr": 25, }
            },
            "30": {
                "500Hz": {"10pr": -1, "50pr": 6, "90pr": 12, },
                "1000Hz": {"10pr": -2, "50pr": 4, "90pr": 11, },
                "2000Hz": {"10pr": -2, "50pr": 4, "90pr": 12, },
                "3000Hz": {"10pr": -2, "50pr": 6, "90pr": 12, },
                "4000Hz": {"10pr": -3, "50pr": 5, "90pr": 12, },
                "6000Hz": {"10pr": 3, "50pr": 13, "90pr": 25, }
            },
            "40": {
                "500Hz": {"10pr": 0, "50pr": 6, "90pr": 13, },
                "1000Hz": {"10pr": -2, "50pr": 6, "90pr": 15, },
                "2000Hz": {"10pr": -2, "50pr": 6, "90pr": 15, },
                "3000Hz": {"10pr": -2, "50pr": 8, "90pr": 19, },
                "4000Hz": {"10pr": -3, "50pr": 8, "90pr": 18, },
                "6000Hz": {"10pr": 4, "50pr": 17, "90pr": 28, }
            },
            "50": {
                "500Hz": {"10pr": 1, "50pr": 8, "90pr": 15, },
                "1000Hz": {"10pr": 1, "50pr": 7, "90pr": 15, },
                "2000Hz": {"10pr": 0, "50pr": 8, "90pr": 18, },
                "3000Hz": {"10pr": 1, "50pr": 11, "90pr": 12, },
                "4000Hz": {"10pr": 4, "50pr": 13, "90pr": 26, },
                "6000Hz": {"10pr": 10, "50pr": 23, "90pr": 44, }
            },
            "60": {
                "500Hz": {"10pr": 3, "50pr": 11, "90pr": 17, },
                "1000Hz": {"10pr": 2, "50pr": 9, "90pr": 17, },
                "2000Hz": {"10pr": 2, "50pr": 13, "90pr": 24, },
                "3000Hz": {"10pr": 5, "50pr": 15, "90pr": 32, },
                "4000Hz": {"10pr": -2, "50pr": 15, "90pr": 41, },
                "6000Hz": {"10pr": 14, "50pr": 28, "90pr": 57, }
            }
        },
        "Female": {
            "20": {
                "500Hz": {"10pr": 2, "50pr": 6, "90pr": 13, },
                "1000Hz": {"10pr": 0, "50pr": 6, "90pr": 12, },
                "2000Hz": {"10pr": -1, "50pr": 5, "90pr": 12, },
                "3000Hz": {"10pr": -2, "50pr": 5, "90pr": 12, },
                "4000Hz": {"10pr": -3, "50pr": 4, "90pr": 12, },
                "6000Hz": {"10pr": 7, "50pr": 13, "90pr": 22, }
            },
            "30": {
                "500Hz": {"10pr": -1, "50pr": 5, "90pr": 11, },
                "1000Hz": {"10pr": -2, "50pr": 4, "90pr": 11, },
                "2000Hz": {"10pr": -2, "50pr": 5, "90pr": 12, },
                "3000Hz": {"10pr": -2, "50pr": 5, "90pr": 13, },
                "4000Hz": {"10pr": -4, "50pr": 3, "90pr": 12, },
                "6000Hz": {"10pr": 3, "50pr": 13, "90pr": 21, }
            },
            "40": {
                "500Hz": {"10pr": 0, "50pr": 6, "90pr": 13, },
                "1000Hz": {"10pr": -2, "50pr": 5, "90pr": 13, },
                "2000Hz": {"10pr": -2, "50pr": 5, "90pr": 13, },
                "3000Hz": {"10pr": -2, "50pr": 6, "90pr": 16, },
                "4000Hz": {"10pr": -4, "50pr": 5, "90pr": 15, },
                "6000Hz": {"10pr": 3, "50pr": 15, "90pr": 27, }
            },
            "50": {
                "500Hz": {"10pr": 2, "50pr": 8, "90pr": 15, },
                "1000Hz": {"10pr": 0, "50pr": 7, "90pr": 17, },
                "2000Hz": {"10pr": 2, "50pr": 9, "90pr": 18, },
                "3000Hz": {"10pr": 1, "50pr": 10, "90pr": 18, },
                "4000Hz": {"10pr": -1, "50pr": 8, "90pr": 17, },
                "6000Hz": {"10pr": 8, "50pr": 19, "90pr": 33, }
            },
            "60": {
                "500Hz": {"10pr": 4, "50pr": 12, "90pr": 25, },
                "1000Hz": {"10pr": 4, "50pr": 12, "90pr": 27, },
                "2000Hz": {"10pr": 5, "50pr": 14, "90pr": 27, },
                "3000Hz": {"10pr": 5, "50pr": 19, "90pr": 34, },
                "4000Hz": {"10pr": 3, "50pr": 19, "90pr": 34, },
                "6000Hz": {"10pr": 11, "50pr": 31, "90pr": 52, }
            }
        }
    }

    ISO_1999_2013_NIPTS_PRED_DICT = {
        "500Hz": {"u": -0.033, "v": 0.110, "L0": 93},
        "1000Hz": {"u": -0.020, "v": 0.070, "L0": 89},
        "2000Hz": {"u": -0.045, "v": 0.066, "L0": 80},
        "3000Hz": {"u": 0.012, "v": 0.037, "L0": 77},
        "4000Hz": {"u": 0.025, "v": 0.025, "L0": 75},
        "6000Hz": {"u": 0.019, "v": 0.024, "L0": 77}
    }

    ISO_1999_2023_NIPTS_PRED_DICT_T1 = {
        "10years":{
            "70dB":{
                "500Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":1,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "4000Hz": {"5pr":1,"10pr":1,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "6000Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
            },
            "75dB":{
                "500Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":2,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "4000Hz": {"5pr":3,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "6000Hz": {"5pr":2,"10pr":1,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
            },
            "80dB":{
                "500Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":1,"10pr":1,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":3,"10pr":2,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":7,"10pr":6,"25pr":4,"50pr":3,"75pr":1,"90%":0,"95%":0},
                "4000Hz": {"5pr":13,"10pr":11,"25pr":7,"50pr":4,"75pr":1,"90%":0,"95%":0},
                "6000Hz": {"5pr":7,"10pr":6,"25pr":4,"50pr":2,"75pr":1,"90%":0,"95%":0},
            },
            "85dB":{
                "500Hz": {"5pr":2,"10pr":2,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":3,"10pr":2,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":8,"10pr":6,"25pr":3,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":12,"10pr":10,"25pr":7,"50pr":5,"75pr":3,"90%":2,"95%":1},
                "4000Hz": {"5pr":19,"10pr":16,"25pr":11,"50pr":7,"75pr":4,"90%":2,"95%":0},
                "6000Hz": {"5pr":14,"10pr":11,"25pr":8,"50pr":5,"75pr":3,"90%":1,"95%":0},
            },
            "90dB":{
                "500Hz": {"5pr":4,"10pr":3,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":5,"10pr":3,"25pr":2,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":12,"10pr":9,"25pr":6,"50pr":3,"75pr":1,"90%":0,"95%":0},
                "3000Hz": {"5pr":24,"10pr":20,"25pr":15,"50pr":10,"75pr":6,"90%":4,"95%":3},
                "4000Hz": {"5pr":27,"10pr":23,"25pr":18,"50pr":13,"75pr":9,"90%":6,"95%":5},
                "6000Hz": {"5pr":22,"10pr":19,"25pr":14,"50pr":9,"75pr":5,"90%":2,"95%":0},
            },
            "95dB":{
                "500Hz": {"5pr":5,"10pr":4,"25pr":2,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":8,"10pr":7,"25pr":4,"50pr":3,"75pr":2,"90%":1,"95%":1},
                "2000Hz": {"5pr":20,"10pr":16,"25pr":11,"50pr":6,"75pr":3,"90%":0,"95%":0},
                "3000Hz": {"5pr":36,"10pr":31,"25pr":24,"50pr":17,"75pr":12,"90%":8,"95%":6},
                "4000Hz": {"5pr":37,"10pr":33,"25pr":27,"50pr":21,"75pr":17,"90%":13,"95%":11},
                "6000Hz": {"5pr":32,"10pr":28,"25pr":21,"50pr":14,"75pr":9,"90%":4,"95%":2},
            },
            "100dB":{
                "500Hz": {"5pr":10,"10pr":8,"25pr":6,"50pr":4,"75pr":3,"90%":2,"95%":1},
                "1000Hz": {"5pr":14,"10pr":12,"25pr":9,"50pr":6,"75pr":4,"90%":3,"95%":2},
                "2000Hz": {"5pr":29,"10pr":24,"25pr":16,"50pr":9,"75pr":5,"90%":1,"95%":0},
                "3000Hz": {"5pr":47,"10pr":42,"25pr":34,"50pr":26,"75pr":20,"90%":15,"95%":13},
                "4000Hz": {"5pr":48,"10pr":44,"25pr":38,"50pr":31,"75pr":26,"90%":21,"95%":19},
                "6000Hz": {"5pr":42,"10pr":37,"25pr":29,"50pr":21,"75pr":14,"90%":9,"95%":6},
            }
        },
        "20years":{
            "70dB":{
                "500Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":2,"10pr":1,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "4000Hz": {"5pr":2,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "6000Hz": {"5pr":2,"10pr":1,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
            },
            "75dB":{
                "500Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":1,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":2,"10pr":1,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":3,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "4000Hz": {"5pr":4,"10pr":3,"25pr":2,"50pr":1,"75pr":1,"90%":0,"95%":0},
                "6000Hz": {"5pr":3,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
            },
            "80dB":{
                "500Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":1,"10pr":1,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":3,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":8,"10pr":6,"25pr":4,"50pr":3,"75pr":3,"90%":2,"95%":2},
                "4000Hz": {"5pr":14,"10pr":11,"25pr":8,"50pr":5,"75pr":2,"90%":1,"95%":0},
                "6000Hz": {"5pr":7,"10pr":6,"25pr":4,"50pr":3,"75pr":2,"90%":2,"95%":2},
            },
            "85dB":{
                "500Hz": {"5pr":2,"10pr":2,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":3,"10pr":3,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":8,"10pr":6,"25pr":4,"50pr":2,"75pr":1,"90%":1,"95%":1},
                "3000Hz": {"5pr":13,"10pr":11,"25pr":8,"50pr":6,"75pr":5,"90%":4,"95%":3},
                "4000Hz": {"5pr":21,"10pr":17,"25pr":13,"50pr":9,"75pr":6,"90%":4,"95%":3},
                "6000Hz": {"5pr":15,"10pr":12,"25pr":9,"50pr":6,"75pr":4,"90%":3,"95%":2},
            },
            "90dB":{
                "500Hz": {"5pr":4,"10pr":3,"25pr":2,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":5,"10pr":5,"25pr":2,"50pr":1,"75pr":1,"90%":0,"95%":0},
                "2000Hz": {"5pr":14,"10pr":11,"25pr":8,"50pr":4,"75pr":3,"90%":2,"95%":1},
                "3000Hz": {"5pr":27,"10pr":23,"25pr":17,"50pr":12,"75pr":9,"90%":7,"95%":5},
                "4000Hz": {"5pr":30,"10pr":26,"25pr":20,"50pr":15,"75pr":11,"90%":8,"95%":7},
                "6000Hz": {"5pr":25,"10pr":21,"25pr":16,"50pr":11,"75pr":7,"90%":4,"95%":3},
            },
            "95dB":{
                "500Hz": {"5pr":6,"10pr":5,"25pr":3,"50pr":1,"75pr":1,"90%":0,"95%":0},
                "1000Hz": {"5pr":10,"10pr":8,"25pr":6,"50pr":4,"75pr":3,"90%":2,"95%":2},
                "2000Hz": {"5pr":24,"10pr":20,"25pr":15,"50pr":9,"75pr":6,"90%":4,"95%":3},
                "3000Hz": {"5pr":40,"10pr":35,"25pr":28,"50pr":21,"75pr":15,"90%":11,"95%":10},
                "4000Hz": {"5pr":41,"10pr":37,"25pr":31,"50pr":25,"75pr":20,"90%":16,"95%":14},
                "6000Hz": {"5pr":36,"10pr":31,"25pr":24,"50pr":17,"75pr":11,"90%":7,"95%":5},
            },
            "100dB":{
                "500Hz": {"5pr":12,"10pr":10,"25pr":8,"50pr":5,"75pr":4,"90%":3,"95%":2},
                "1000Hz": {"5pr":18,"10pr":15,"25pr":11,"50pr":8,"75pr":6,"90%":5,"95%":4},
                "2000Hz": {"5pr":36,"10pr":31,"25pr":23,"50pr":15,"75pr":10,"90%":6,"95%":4},
                "3000Hz": {"5pr":52,"10pr":48,"25pr":40,"50pr":32,"75pr":25,"90%":20,"95%":17},
                "4000Hz": {"5pr":51,"10pr":48,"25pr":42,"50pr":36,"75pr":30,"90%":25,"95%":23},
                "6000Hz": {"5pr":48,"10pr":43,"25pr":35,"50pr":26,"75pr":19,"90%":13,"95%":10},
            }
        },
        "30years":{
            "70dB":{
                "500Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":2,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "4000Hz": {"5pr":3,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "6000Hz": {"5pr":2,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
            },
            "75dB":{
                "500Hz": {"5pr":1,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":2,"10pr":1,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":4,"10pr":3,"25pr":2,"50pr":1,"75pr":1,"90%":0,"95%":0},
                "4000Hz": {"5pr":6,"10pr":4,"25pr":3,"50pr":1,"75pr":1,"90%":0,"95%":0},
                "6000Hz": {"5pr":4,"10pr":3,"25pr":2,"50pr":1,"75pr":1,"90%":0,"95%":0},
            },
            "80dB":{
                "500Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":2,"10pr":2,"25pr":1,"50pr":1,"75pr":1,"90%":1,"95%":1},
                "3000Hz": {"5pr":8,"10pr":6,"25pr":5,"50pr":4,"75pr":4,"90%":4,"95%":4},
                "4000Hz": {"5pr":15,"10pr":12,"25pr":8,"50pr":6,"75pr":4,"90%":2,"95%":2},
                "6000Hz": {"5pr":7,"10pr":6,"25pr":4,"50pr":4,"75pr":4,"90%":4,"95%":4},
            },
            "85dB":{
                "500Hz": {"5pr":2,"10pr":1,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":3,"10pr":3,"25pr":2,"50pr":1,"75pr":1,"90%":1,"95%":1},
                "2000Hz": {"5pr":9,"10pr":7,"25pr":4,"50pr":2,"75pr":2,"90%":2,"95%":2},
                "3000Hz": {"5pr":14,"10pr":12,"25pr":10,"50pr":8,"75pr":6,"90%":6,"95%":6},
                "4000Hz": {"5pr":23,"10pr":19,"25pr":14,"50pr":10,"75pr":7,"90%":6,"95%":5},
                "6000Hz": {"5pr":16,"10pr":14,"25pr":10,"50pr":8,"75pr":6,"90%":5,"95%":4},
            },
            "90dB":{
                "500Hz": {"5pr":4,"10pr":3,"25pr":2,"50pr":1,"75pr":1,"90%":1,"95%":1},
                "1000Hz": {"5pr":6,"10pr":4,"25pr":3,"50pr":1,"75pr":1,"90%":1,"95%":1},
                "2000Hz": {"5pr":16,"10pr":13,"25pr":9,"50pr":6,"75pr":5,"90%":4,"95%":3},
                "3000Hz": {"5pr":29,"10pr":25,"25pr":20,"50pr":15,"75pr":11,"90%":9,"95%":8},
                "4000Hz": {"5pr":33,"10pr":29,"25pr":23,"50pr":18,"75pr":14,"90%":11,"95%":10},
                "6000Hz": {"5pr":28,"10pr":24,"25pr":18,"50pr":13,"75pr":9,"90%":6,"95%":5},
            },
            "95dB":{
                "500Hz": {"5pr":7,"10pr":5,"25pr":3,"50pr":2,"75pr":1,"90%":1,"95%":0},
                "1000Hz": {"5pr":12,"10pr":10,"25pr":7,"50pr":5,"75pr":4,"90%":3,"95%":3},
                "2000Hz": {"5pr":29,"10pr":25,"25pr":19,"50pr":13,"75pr":10,"90%":7,"95%":6},
                "3000Hz": {"5pr":45,"10pr":40,"25pr":32,"50pr":24,"75pr":19,"90%":15,"95%":13},
                "4000Hz": {"5pr":45,"10pr":41,"25pr":35,"50pr":28,"75pr":23,"90%":19,"95%":18},
                "6000Hz": {"5pr":40,"10pr":35,"25pr":28,"50pr":20,"75pr":14,"90%":9,"95%":7},
            },
            "100dB":{
                "500Hz": {"5pr":14,"10pr":12,"25pr":9,"50pr":6,"75pr":5,"90%":4,"95%":3},
                "1000Hz": {"5pr":21,"10pr":18,"25pr":14,"50pr":10,"75pr":8,"90%":6,"95%":5},
                "2000Hz": {"5pr":42,"10pr":37,"25pr":29,"50pr":21,"75pr":16,"90%":12,"95%":10},
                "3000Hz": {"5pr":58,"10pr":53,"25pr":45,"50pr":37,"75pr":30,"90%":25,"95%":22},
                "4000Hz": {"5pr":56,"10pr":52,"25pr":46,"50pr":40,"75pr":34,"90%":29,"95%":27},
                "6000Hz": {"5pr":55,"10pr":49,"25pr":41,"50pr":32,"75pr":24,"90%":18,"95%":15},
            }
        },
        "40years":{
            "70dB":{
                "500Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":1,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":1,"10pr":1,"25pr":1,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":3,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "4000Hz": {"5pr":4,"10pr":3,"25pr":2,"50pr":1,"75pr":1,"90%":0,"95%":0},
                "6000Hz": {"5pr":3,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
            },
            "75dB":{
                "500Hz": {"5pr":1,"10pr":0,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "1000Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":0,"90%":0,"95%":0},
                "2000Hz": {"5pr":3,"10pr":2,"25pr":1,"50pr":1,"75pr":0,"90%":0,"95%":0},
                "3000Hz": {"5pr":5,"10pr":4,"25pr":2,"50pr":1,"75pr":1,"90%":0,"95%":0},
                "4000Hz": {"5pr":7,"10pr":5,"25pr":3,"50pr":2,"75pr":1,"90%":1,"95%":0},
                "6000Hz": {"5pr":6,"10pr":4,"25pr":2,"50pr":1,"75pr":1,"90%":0,"95%":0},
            },
            "80dB":{
                "500Hz": {"5pr":0,"10pr":0,"25pr":0,"50pr":0,"75pr":1,"90%":1,"95%":1},
                "1000Hz": {"5pr":1,"10pr":1,"25pr":0,"50pr":0,"75pr":1,"90%":1,"95%":1},
                "2000Hz": {"5pr":1,"10pr":1,"25pr":1,"50pr":1,"75pr":2,"90%":2,"95%":2},
                "3000Hz": {"5pr":7,"10pr":6,"25pr":5,"50pr":5,"75pr":5,"90%":6,"95%":6},
                "4000Hz": {"5pr":16,"10pr":13,"25pr":9,"50pr":7,"75pr":5,"90%":4,"95%":4},
                "6000Hz": {"5pr":7,"10pr":6,"25pr":5,"50pr":5,"75pr":5,"90%":6,"95%":6},
            },
            "85dB":{
                "500Hz": {"5pr":2,"10pr":1,"25pr":1,"50pr":1,"75pr":1,"90%":1,"95%":1},
                "1000Hz": {"5pr":4,"10pr":3,"25pr":2,"50pr":1,"75pr":1,"90%":1,"95%":1},
                "2000Hz": {"5pr":8,"10pr":7,"25pr":4,"50pr":3,"75pr":3,"90%":3,"95%":3},
                "3000Hz": {"5pr":16,"10pr":13,"25pr":11,"50pr":9,"75pr":8,"90%":8,"95%":8},
                "4000Hz": {"5pr":25,"10pr":21,"25pr":16,"50pr":12,"75pr":9,"90%":8,"95%":7},
                "6000Hz": {"5pr":18,"10pr":15,"25pr":12,"50pr":9,"75pr":8,"90%":7,"95%":7},
            },
            "90dB":{
                "500Hz": {"5pr":4,"10pr":3,"25pr":2,"50pr":1,"75pr":1,"90%":1,"95%":1},
                "1000Hz": {"5pr":6,"10pr":5,"25pr":3,"50pr":2,"75pr":1,"90%":1,"95%":1},
                "2000Hz": {"5pr":18,"10pr":15,"25pr":11,"50pr":8,"75pr":6,"90%":6,"95%":5},
                "3000Hz": {"5pr":33,"10pr":28,"25pr":23,"50pr":17,"75pr":14,"90%":12,"95%":11},
                "4000Hz": {"5pr":37,"10pr":33,"25pr":26,"50pr":20,"75pr":16,"90%":14,"95%":13},
                "6000Hz": {"5pr":32,"10pr":27,"25pr":21,"50pr":15,"75pr":11,"90%":9,"95%":7},
            },
            "95dB":{
                "500Hz": {"5pr":7,"10pr":6,"25pr":4,"50pr":2,"75pr":1,"90%":1,"95%":1},
                "1000Hz": {"5pr":13,"10pr":11,"25pr":8,"50pr":6,"75pr":5,"90%":4,"95%":4},
                "2000Hz": {"5pr":44,"10pr":29,"25pr":23,"50pr":17,"75pr":13,"90%":11,"95%":10},
                "3000Hz": {"5pr":50,"10pr":45,"25pr":36,"50pr":28,"75pr":23,"90%":19,"95%":17},
                "4000Hz": {"5pr":50,"10pr":46,"25pr":39,"50pr":33,"75pr":27,"90%":23,"95%":21},
                "6000Hz": {"5pr":46,"10pr":41,"25pr":32,"50pr":24,"75pr":17,"90%":13,"95%":10},
            },
            "100dB":{
                "500Hz": {"5pr":16,"10pr":13,"25pr":10,"50pr":8,"75pr":6,"90%":5,"95%":4},
                "1000Hz": {"5pr":23,"10pr":20,"25pr":16,"50pr":12,"75pr":10,"90%":8,"95%":8},
                "2000Hz": {"5pr":49,"10pr":44,"25pr":36,"50pr":28,"75pr":22,"90%":17,"95%":15},
                "3000Hz": {"5pr":65,"10pr":60,"25pr":51,"50pr":42,"75pr":35,"90%":30,"95%":27},
                "4000Hz": {"5pr":62,"10pr":58,"25pr":52,"50pr":45,"75pr":39,"90%":34,"95%":31},
                "6000Hz": {"5pr":64,"10pr":58,"25pr":48,"50pr":38,"75pr":30,"90%":23,"95%":20},
            }
        }
    }

    ISO_1999_2023_NIPTS_PRED_DICT_B6 = {
        "Female": {
            "20": {
                "500Hz": {"10pr": 12, "50pr": 6, "90pr": 2},
                "1000Hz": {"10pr": 12, "50pr": 6, "90pr": 0},
                "2000Hz": {"10pr": 12, "50pr": 5, "90pr": -1},
                "3000Hz": {"10pr": 12, "50pr": 5, "90pr": -2},
                "4000Hz": {"10pr": 12, "50pr": 4, "90pr": -3},
                "6000Hz": {"10pr": 22, "50pr": 13, "90pr": 7}
            },
            "30": {
                "500Hz": {"10pr": 11, "50pr": 5, "90pr": -1},
                "1000Hz": {"10pr": 11, "50pr": 4, "90pr": -2},
                "2000Hz": {"10pr": 12, "50pr": 5, "90pr": -2},
                "3000Hz": {"10pr": 13, "50pr": 5, "90pr": -2},
                "4000Hz": {"10pr": 12, "50pr": 3, "90pr": -4},
                "6000Hz": {"10pr": 21, "50pr": 13, "90pr": 3}
            },
            "40": {
                "500Hz": {"10pr": 13, "50pr": 6, "90pr": 0},
                "1000Hz": {"10pr": 13, "50pr": 5, "90pr": -2},
                "2000Hz": {"10pr": 13, "50pr": 5, "90pr": -2},
                "3000Hz": {"10pr": 16, "50pr": 6, "90pr": -2},
                "4000Hz": {"10pr": 15, "50pr": 5, "90pr": -4},
                "6000Hz": {"10pr": 27, "50pr": 15, "90pr": 3}
            },
            "50": {
                "500Hz": {"10pr": 15, "50pr": 8, "90pr": 2},
                "1000Hz": {"10pr": 17, "50pr": 7, "90pr": 0},
                "2000Hz": {"10pr": 18, "50pr": 9, "90pr": 2},
                "3000Hz": {"10pr": 18, "50pr": 10, "90pr": 1},
                "4000Hz": {"10pr": 17, "50pr": 8, "90pr": -1},
                "6000Hz": {"10pr": 33, "50pr": 19, "90pr": 8}
            },
            "60": {
                "500Hz": {"10pr": 25, "50pr": 12, "90pr": 4},
                "1000Hz": {"10pr": 27, "50pr": 12, "90pr": 4},
                "2000Hz": {"10pr": 27, "50pr": 14, "90pr": 5},
                "3000Hz": {"10pr": 34, "50pr": 19, "90pr": 5},
                "4000Hz": {"10pr": 34, "50pr": 19, "90pr": 3},
                "6000Hz": {"10pr": 52, "50pr": 31, "90pr": 11}
            }
        },
        "Male": {
            "20": {
                "500Hz": {"10pr": 10, "50pr": 6, "90pr": 3},
                "1000Hz": {"10pr": 12, "50pr": 6, "90pr": 1},
                "2000Hz": {"10pr": 11, "50pr": 4, "90pr": -2},
                "3000Hz": {"10pr": 12, "50pr": 5, "90pr": 1},
                "4000Hz": {"10pr": 12, "50pr": 6, "90pr": -3},
                "6000Hz": {"10pr": 25, "50pr": 12, "90pr": 3}
            },
            "30": {
                "500Hz": {"10pr": 12, "50pr": 6, "90pr": -1},
                "1000Hz": {"10pr": 11, "50pr": 4, "90pr": -2},
                "2000Hz": {"10pr": 12, "50pr": 4, "90pr": -2},
                "3000Hz": {"10pr": 12, "50pr": 6, "90pr": -2},
                "4000Hz": {"10pr": 12, "50pr": 5, "90pr": -3},
                "6000Hz": {"10pr": 25, "50pr": 13, "90pr": 4}
            },
            "40": {
                "500Hz": {"10pr": 13, "50pr": 6, "90pr": 0},
                "1000Hz": {"10pr": None, "50pr": 6, "90pr": -2},
                "2000Hz": {"10pr": 15, "50pr": 6, "90pr": -2},
                "3000Hz": {"10pr": 19, "50pr": 8, "90pr": -2},
                "4000Hz": {"10pr": 18, "50pr": 8, "90pr": -3},
                "6000Hz": {"10pr": 28, "50pr": 17, "90pr": 4}
            },
            "50": {
                "500Hz": {"10pr": 15, "50pr": 8, "90pr": 1},
                "1000Hz": {"10pr": 15, "50pr": 7, "90pr": 1},
                "2000Hz": {"10pr": 18, "50pr": 8, "90pr": 0},
                "3000Hz": {"10pr": 12, "50pr": 11, "90pr": 1},
                "4000Hz": {"10pr": 26, "50pr": 13, "90pr": 4},
                "6000Hz": {"10pr": 44, "50pr": 23, "90pr": 10}
            },
            "60": {
                "500Hz": {"10pr": 17, "50pr": 11, "90pr": 3},
                "1000Hz": {"10pr": 17, "50pr": 9, "90pr": 2},
                "2000Hz": {"10pr": 24, "50pr": 13, "90pr": 2},
                "3000Hz": {"10pr": 32, "50pr": 15, "90pr": 5},
                "4000Hz": {"10pr": 41, "50pr": 15, "90pr": -2},
                "6000Hz": {"10pr": 57, "50pr": 28, "90pr": 14}
            }
        },
    }

    BASELINE_NOISE_KURTOSIS = 3
