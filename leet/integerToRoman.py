from math import floor
class Solution:

    def intToRoman(self, num: int) -> str:
        converter = {
            "1000": "M",
            "900": "CM",
            "500": "D",            
            "400": "CD",            
            "100": "C",
            "90": "XC",
            "50": "L",
            "40": "XL",
            "10": "X",
            "9": "IX",
            "5": "V",
            "4": "IV",
            "1": "I"
        }
        final_str = ""
        for amount, symbol in converter.items():
            if int(amount) > num:
                continue
            symbol_count = floor(num / int(amount))
            final_str += symbol * symbol_count
            num = num - (int(amount) * symbol_count)
        return final_str