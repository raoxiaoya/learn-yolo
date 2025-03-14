import random
import string


def generate_random_string(length):
    """Generate a random string of specific length."""
    # 字符集包括所有字母（大写和小写）和数字
    characters = string.ascii_letters + string.digits
    # 使用random.choices从给定字符集中随机选择字符，k参数指定了要选择多少个字符
    random_string = ''.join(random.choices(characters, k=length))
    return random_string
