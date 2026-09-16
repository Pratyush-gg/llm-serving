"""
Dataset preparation script for Code adapter.
Generates 600 training tasks and 60 held-out evaluation tasks with unit-test assertions.
Every held-out sample is dynamically verified to pass all assertions against its gold solution.
"""

import json
import os
import textwrap

def verify_code_and_assertions(func_code: str, assertions: str) -> bool:
    """Safely verify that the gold solution passes all its assertions."""
    scope = {}
    try:
        exec(func_code, scope)
        exec(assertions, scope)
        return True
    except Exception as e:
        print(f"Assertion failed: {e}\nCode:\n{func_code}\nAssertions:\n{assertions}")
        return False

# Comprehensive bank of distinct algorithmic, utility, and data-processing tasks
# Each task has: (func_name, docstring_and_signature, gold_implementation, assertions)
CODE_TASKS = [
    # 1. String Manipulation
    (
        "reverse_string",
        "def reverse_string(s: str) -> str:\n    \"\"\"Return the reversed string.\"\"\"",
        "def reverse_string(s: str) -> str:\n    return s[::-1]",
        "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nassert reverse_string('Python') == 'nohtyP'"
    ),
    (
        "is_palindrome",
        "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is a palindrome, ignoring case and non-alphanumeric.\"\"\"",
        "def is_palindrome(s: str) -> bool:\n    filtered = [c.lower() for c in s if c.isalnum()]\n    return filtered == filtered[::-1]",
        "assert is_palindrome('A man, a plan, a canal: Panama') == True\nassert is_palindrome('race a car') == False\nassert is_palindrome('') == True"
    ),
    (
        "count_vowels",
        "def count_vowels(s: str) -> int:\n    \"\"\"Return count of vowels (a, e, i, o, u) case-insensitive.\"\"\"",
        "def count_vowels(s: str) -> int:\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        "assert count_vowels('Antigravity') == 4\nassert count_vowels('xyz') == 0\nassert count_vowels('AEIOU') == 5"
    ),
    (
        "capitalize_words",
        "def capitalize_words(s: str) -> str:\n    \"\"\"Capitalize first letter of each word.\"\"\"",
        "def capitalize_words(s: str) -> str:\n    return ' '.join(word.capitalize() for word in s.split(' '))",
        "assert capitalize_words('hello world') == 'Hello World'\nassert capitalize_words('deep learning') == 'Deep Learning'"
    ),
    (
        "truncate_string",
        "def truncate_string(s: str, max_len: int) -> str:\n    \"\"\"Truncate string to max_len, appending '...' if truncated.\"\"\"",
        "def truncate_string(s: str, max_len: int) -> str:\n    if len(s) <= max_len:\n        return s\n    return s[:max_len - 3] + '...'",
        "assert truncate_string('Hello World', 8) == 'Hello...'\nassert truncate_string('Hi', 5) == 'Hi'\nassert truncate_string('Testing 123', 6) == 'Tes...'"
    ),
    (
        "to_snake_case",
        "def to_snake_case(s: str) -> str:\n    \"\"\"Convert camelCase or PascalCase to snake_case.\"\"\"",
        "def to_snake_case(s: str) -> str:\n    res = []\n    for i, c in enumerate(s):\n        if c.isupper() and i > 0 and s[i-1] != '_':\n            res.append('_')\n        res.append(c.lower())\n    return ''.join(res)",
        "assert to_snake_case('camelCase') == 'camel_case'\nassert to_snake_case('PascalCase') == 'pascal_case'\nassert to_snake_case('simple') == 'simple'"
    ),
    (
        "to_camel_case",
        "def to_camel_case(s: str) -> str:\n    \"\"\"Convert snake_case to camelCase.\"\"\"",
        "def to_camel_case(s: str) -> str:\n    parts = s.split('_')\n    return parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])",
        "assert to_camel_case('snake_case') == 'snakeCase'\nassert to_camel_case('get_user_by_id') == 'getUserById'"
    ),
    (
        "remove_duplicate_words",
        "def remove_duplicate_words(s: str) -> str:\n    \"\"\"Remove consecutive duplicate words from string.\"\"\"",
        "def remove_duplicate_words(s: str) -> str:\n    words = s.split()\n    if not words: return ''\n    res = [words[0]]\n    for w in words[1:]:\n        if w != res[-1]:\n            res.append(w)\n    return ' '.join(res)",
        "assert remove_duplicate_words('hello hello world world') == 'hello world'\nassert remove_duplicate_words('a b a') == 'a b a'"
    ),
    (
        "count_consonants",
        "def count_consonants(s: str) -> int:\n    \"\"\"Count consonants in string.\"\"\"",
        "def count_consonants(s: str) -> int:\n    return sum(1 for c in s.lower() if c.isalpha() and c not in 'aeiou')",
        "assert count_consonants('hello') == 3\nassert count_consonants('aeiou') == 0\nassert count_consonants('Python') == 5"
    ),
    (
        "mask_email",
        "def mask_email(email: str) -> str:\n    \"\"\"Mask username of email, showing only first and last character.\"\"\"",
        "def mask_email(email: str) -> str:\n    user, domain = email.split('@')\n    if len(user) <= 2:\n        masked = user[0] + '*'\n    else:\n        masked = user[0] + '*' * (len(user) - 2) + user[-1]\n    return f'{masked}@{domain}'",
        "assert mask_email('john.doe@example.com') == 'j******e@example.com'\nassert mask_email('me@test.com') == 'm*@test.com'"
    ),

    # 2. Math & Number Theory
    (
        "is_prime",
        "def is_prime(n: int) -> bool:\n    \"\"\"Check if n is prime.\"\"\"",
        "def is_prime(n: int) -> bool:\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True",
        "assert is_prime(2) == True\nassert is_prime(17) == True\nassert is_prime(4) == False\nassert is_prime(1) == False"
    ),
    (
        "gcd",
        "def gcd(a: int, b: int) -> int:\n    \"\"\"Compute greatest common divisor of a and b.\"\"\"",
        "def gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return a",
        "assert gcd(12, 18) == 6\nassert gcd(101, 10) == 1\nassert gcd(0, 5) == 5"
    ),
    (
        "lcm",
        "def lcm(a: int, b: int) -> int:\n    \"\"\"Compute least common multiple of a and b.\"\"\"",
        "def lcm(a: int, b: int) -> int:\n    def _gcd(x, y):\n        while y: x, y = y, x % y\n        return x\n    return abs(a * b) // _gcd(a, b)",
        "assert lcm(4, 6) == 12\nassert lcm(5, 7) == 35"
    ),
    (
        "fibonacci",
        "def fibonacci(n: int) -> int:\n    \"\"\"Return n-th Fibonacci number (0-indexed: 0, 1, 1, 2, 3, ...).\"\"\"",
        "def fibonacci(n: int) -> int:\n    if n <= 0: return 0\n    if n == 1: return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
        "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(6) == 8\nassert fibonacci(10) == 55"
    ),
    (
        "factorial",
        "def factorial(n: int) -> int:\n    \"\"\"Compute factorial of non-negative integer n.\"\"\"",
        "def factorial(n: int) -> int:\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res",
        "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(3) == 6"
    ),
    (
        "is_power_of_two",
        "def is_power_of_two(n: int) -> bool:\n    \"\"\"Check if integer n is a power of two.\"\"\"",
        "def is_power_of_two(n: int) -> bool:\n    return n > 0 and (n & (n - 1)) == 0",
        "assert is_power_of_two(1) == True\nassert is_power_of_two(16) == True\nassert is_power_of_two(18) == False\nassert is_power_of_two(0) == False"
    ),
    (
        "sum_of_digits",
        "def sum_of_digits(n: int) -> int:\n    \"\"\"Return sum of decimal digits of absolute value of n.\"\"\"",
        "def sum_of_digits(n: int) -> int:\n    return sum(int(d) for d in str(abs(n)))",
        "assert sum_of_digits(1234) == 10\nassert sum_of_digits(-505) == 10\nassert sum_of_digits(0) == 0"
    ),
    (
        "collatz_length",
        "def collatz_length(n: int) -> int:\n    \"\"\"Return number of steps to reach 1 in Collatz conjecture.\"\"\"",
        "def collatz_length(n: int) -> int:\n    steps = 0\n    while n > 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        steps += 1\n    return steps",
        "assert collatz_length(1) == 0\nassert collatz_length(6) == 8"
    ),
    (
        "celsius_to_fahrenheit",
        "def celsius_to_fahrenheit(c: float) -> float:\n    \"\"\"Convert Celsius to Fahrenheit rounded to 2 decimals.\"\"\"",
        "def celsius_to_fahrenheit(c: float) -> float:\n    return round(c * 9/5 + 32, 2)",
        "assert celsius_to_fahrenheit(0) == 32.0\nassert celsius_to_fahrenheit(100) == 212.0\nassert celsius_to_fahrenheit(-40) == -40.0"
    ),
    (
        "fahrenheit_to_celsius",
        "def fahrenheit_to_celsius(f: float) -> float:\n    \"\"\"Convert Fahrenheit to Celsius rounded to 2 decimals.\"\"\"",
        "def fahrenheit_to_celsius(f: float) -> float:\n    return round((f - 32) * 5/9, 2)",
        "assert fahrenheit_to_celsius(32.0) == 0.0\nassert fahrenheit_to_celsius(212.0) == 100.0"
    ),

    # 3. Lists & Sequences
    (
        "flatten_list",
        "def flatten_list(nested: list) -> list:\n    \"\"\"Flatten arbitrary nested list of integers.\"\"\"",
        "def flatten_list(nested: list) -> list:\n    res = []\n    for item in nested:\n        if isinstance(item, list):\n            res.extend(flatten_list(item))\n        else:\n            res.append(item)\n    return res",
        "assert flatten_list([1, [2, [3, 4], 5]]) == [1, 2, 3, 4, 5]\nassert flatten_list([]) == []\nassert flatten_list([[1], [2]]) == [1, 2]"
    ),
    (
        "deduplicate_preserve_order",
        "def deduplicate_preserve_order(lst: list) -> list:\n    \"\"\"Remove duplicates while preserving original order.\"\"\"",
        "def deduplicate_preserve_order(lst: list) -> list:\n    seen = set()\n    res = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            res.append(x)\n    return res",
        "assert deduplicate_preserve_order([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]\nassert deduplicate_preserve_order(['a', 'b', 'a']) == ['a', 'b']"
    ),
    (
        "chunk_list",
        "def chunk_list(lst: list, size: int) -> list:\n    \"\"\"Split list into chunks of given size.\"\"\"",
        "def chunk_list(lst: list, size: int) -> list:\n    return [lst[i:i + size] for i in range(0, len(lst), size)]",
        "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunk_list([], 3) == []"
    ),
    (
        "running_sum",
        "def running_sum(nums: list[int]) -> list[int]:\n    \"\"\"Return running sum of elements.\"\"\"",
        "def running_sum(nums: list[int]) -> list[int]:\n    total = 0\n    res = []\n    for n in nums:\n        total += n\n        res.append(total)\n    return res",
        "assert running_sum([1, 2, 3, 4]) == [1, 3, 6, 10]\nassert running_sum([]) == []\nassert running_sum([5]) == [5]"
    ),
    (
        "rotate_list",
        "def rotate_list(lst: list, k: int) -> list:\n    \"\"\"Rotate list to the right by k positions.\"\"\"",
        "def rotate_list(lst: list, k: int) -> list:\n    if not lst: return []\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k] if k != 0 else list(lst)",
        "assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]\nassert rotate_list([1, 2], 0) == [1, 2]"
    ),
    (
        "find_missing_number",
        "def find_missing_number(nums: list[int]) -> int:\n    \"\"\"Given list containing n distinct numbers in range [0, n], return missing number.\"\"\"",
        "def find_missing_number(nums: list[int]) -> int:\n    n = len(nums)\n    return n * (n + 1) // 2 - sum(nums)",
        "assert find_missing_number([3, 0, 1]) == 2\nassert find_missing_number([0, 1]) == 2\nassert find_missing_number([9,6,4,2,3,5,7,0,1]) == 8"
    ),
    (
        "two_sum",
        "def two_sum(nums: list[int], target: int) -> list[int]:\n    \"\"\"Return indices of two numbers that add up to target.\"\"\"",
        "def two_sum(nums: list[int], target: int) -> list[int]:\n    lookup = {}\n    for i, n in enumerate(nums):\n        diff = target - n\n        if diff in lookup:\n            return [lookup[diff], i]\n        lookup[n] = i\n    return []",
        "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\nassert two_sum([3, 2, 4], 6) == [1, 2]"
    ),
    (
        "merge_sorted_lists",
        "def merge_sorted_lists(l1: list[int], l2: list[int]) -> list[int]:\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"",
        "def merge_sorted_lists(l1: list[int], l2: list[int]) -> list[int]:\n    res = []\n    i, j = 0, 0\n    while i < len(l1) and j < len(l2):\n        if l1[i] <= l2[j]:\n            res.append(l1[i])\n            i += 1\n        else:\n            res.append(l2[j])\n            j += 1\n    res.extend(l1[i:])\n    res.extend(l2[j:])\n    return res",
        "assert merge_sorted_lists([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted_lists([], [1, 2]) == [1, 2]"
    ),
    (
        "max_subarray_sum",
        "def max_subarray_sum(nums: list[int]) -> int:\n    \"\"\"Return maximum sum of contiguous subarray (Kadane's algorithm).\"\"\"",
        "def max_subarray_sum(nums: list[int]) -> int:\n    if not nums: return 0\n    max_so_far = max_ending_here = nums[0]\n    for x in nums[1:]:\n        max_ending_here = max(x, max_ending_here + x)\n        max_so_far = max(max_so_far, max_ending_here)\n    return max_so_far",
        "assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\nassert max_subarray_sum([1]) == 1\nassert max_subarray_sum([-1]) == -1"
    ),
    (
        "interleave_lists",
        "def interleave_lists(l1: list, l2: list) -> list:\n    \"\"\"Interleave two lists alternately; append remaining.\"\"\"",
        "def interleave_lists(l1: list, l2: list) -> list:\n    res = []\n    i, j = 0, 0\n    while i < len(l1) or j < len(l2):\n        if i < len(l1):\n            res.append(l1[i])\n            i += 1\n        if j < len(l2):\n            res.append(l2[j])\n            j += 1\n    return res",
        "assert interleave_lists([1, 2, 3], ['a', 'b']) == [1, 'a', 2, 'b', 3]\nassert interleave_lists([], [1, 2]) == [1, 2]"
    ),

    # 4. Dictionaries & Data Structures
    (
        "invert_dict",
        "def invert_dict(d: dict) -> dict:\n    \"\"\"Invert dictionary mapping (values become keys, keys become values).\"\"\"",
        "def invert_dict(d: dict) -> dict:\n    return {v: k for k, v in d.items()}",
        "assert invert_dict({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}\nassert invert_dict({}) == {}"
    ),
    (
        "merge_dicts_sum",
        "def merge_dicts_sum(d1: dict, d2: dict) -> dict:\n    \"\"\"Merge two dicts summing values for overlapping keys.\"\"\"",
        "def merge_dicts_sum(d1: dict, d2: dict) -> dict:\n    res = dict(d1)\n    for k, v in d2.items():\n        res[k] = res.get(k, 0) + v\n    return res",
        "assert merge_dicts_sum({'a': 1, 'b': 2}, {'b': 3, 'c': 4}) == {'a': 1, 'b': 5, 'c': 4}"
    ),
    (
        "frequency_counter",
        "def frequency_counter(items: list) -> dict:\n    \"\"\"Return dictionary of item frequencies.\"\"\"",
        "def frequency_counter(items: list) -> dict:\n    freq = {}\n    for x in items:\n        freq[x] = freq.get(x, 0) + 1\n    return freq",
        "assert frequency_counter(['apple', 'banana', 'apple']) == {'apple': 2, 'banana': 1}\nassert frequency_counter([]) == {}"
    ),
    (
        "filter_dict_by_threshold",
        "def filter_dict_by_threshold(d: dict, min_val: float) -> dict:\n    \"\"\"Return dict with only items where value >= min_val.\"\"\"",
        "def filter_dict_by_threshold(d: dict, min_val: float) -> dict:\n    return {k: v for k, v in d.items() if v >= min_val}",
        "assert filter_dict_by_threshold({'a': 10, 'b': 5, 'c': 15}, 8) == {'a': 10, 'c': 15}"
    ),
    (
        "deep_get",
        "def deep_get(d: dict, keys: list[str], default=None):\n    \"\"\"Retrieve value in nested dict given a list of keys.\"\"\"",
        "def deep_get(d: dict, keys: list[str], default=None):\n    curr = d\n    for k in keys:\n        if isinstance(curr, dict) and k in curr:\n            curr = curr[k]\n        else:\n            return default\n    return curr",
        "assert deep_get({'user': {'profile': {'age': 30}}}, ['user', 'profile', 'age']) == 30\nassert deep_get({'user': {}}, ['user', 'profile', 'age'], -1) == -1"
    ),
    (
        "group_by_key",
        "def group_by_key(records: list[dict], key: str) -> dict:\n    \"\"\"Group a list of dicts by the given key.\"\"\"",
        "def group_by_key(records: list[dict], key: str) -> dict:\n    groups = {}\n    for r in records:\n        val = r.get(key)\n        if val not in groups:\n            groups[val] = []\n        groups[val].append(r)\n    return groups",
        "res = group_by_key([{'cat': 'A', 'v': 1}, {'cat': 'B', 'v': 2}, {'cat': 'A', 'v': 3}], 'cat')\nassert len(res['A']) == 2 and len(res['B']) == 1"
    ),

    # 5. Parsing, Validation & Formatting
    (
        "is_valid_brackets",
        "def is_valid_brackets(s: str) -> bool:\n    \"\"\"Check if brackets '()', '[]', '{}' are validly closed.\"\"\"",
        "def is_valid_brackets(s: str) -> bool:\n    stack = []\n    mapping = {')': '(', ']': '[', '}': '{'}\n    for c in s:\n        if c in mapping.values():\n            stack.append(c)\n        elif c in mapping:\n            if not stack or stack.pop() != mapping[c]:\n                return False\n    return len(stack) == 0",
        "assert is_valid_brackets('()[]{}') == True\nassert is_valid_brackets('(]') == False\nassert is_valid_brackets('([{}])') == True\nassert is_valid_brackets('(') == False"
    ),
    (
        "is_valid_ipv4",
        "def is_valid_ipv4(ip: str) -> bool:\n    \"\"\"Validate if string is a valid IPv4 address.\"\"\"",
        "def is_valid_ipv4(ip: str) -> bool:\n    parts = ip.split('.')\n    if len(parts) != 4:\n        return False\n    for p in parts:\n        if not p.isdigit() or not 0 <= int(p) <= 255 or (len(p) > 1 and p[0] == '0'):\n            return False\n    return True",
        "assert is_valid_ipv4('192.168.1.1') == True\nassert is_valid_ipv4('256.100.0.1') == False\nassert is_valid_ipv4('192.168.01.1') == False\nassert is_valid_ipv4('192.168.1') == False"
    ),
    (
        "parse_query_string",
        "def parse_query_string(query: str) -> dict:\n    \"\"\"Parse URL query string (e.g. 'a=1&b=two') into a dictionary.\"\"\"",
        "def parse_query_string(query: str) -> dict:\n    if not query:\n        return {}\n    res = {}\n    for pair in query.lstrip('?').split('&'):\n        if '=' in pair:\n            k, v = pair.split('=', 1)\n            res[k] = v\n    return res",
        "assert parse_query_string('name=alice&age=25') == {'name': 'alice', 'age': '25'}\nassert parse_query_string('') == {}"
    ),
    (
        "run_length_encode",
        "def run_length_encode(s: str) -> str:\n    \"\"\"Run-length encode string (e.g. 'AAABBBCC' -> '3A3B2C').\"\"\"",
        "def run_length_encode(s: str) -> str:\n    if not s: return ''\n    res = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            res.append(f'{count}{s[i-1]}')\n            count = 1\n    res.append(f'{count}{s[-1]}')\n    return ''.join(res)",
        "assert run_length_encode('AAABBBCC') == '3A3B2C'\nassert run_length_encode('A') == '1A'\nassert run_length_encode('') == ''"
    ),
    (
        "run_length_decode",
        "def run_length_decode(s: str) -> str:\n    \"\"\"Decode run-length encoded string (e.g. '3A3B2C' -> 'AAABBBCC').\"\"\"",
        "def run_length_decode(s: str) -> str:\n    import re\n    matches = re.findall(r'(\\d+)([a-zA-Z])', s)\n    return ''.join(int(cnt) * char for cnt, char in matches)",
        "assert run_length_decode('3A3B2C') == 'AAABBBCC'\nassert run_length_decode('1A') == 'A'"
    ),
    (
        "format_currency",
        "def format_currency(cents: int) -> str:\n    \"\"\"Format amount in integer cents to standard USD format (e.g. 1250 -> '$12.50').\"\"\"",
        "def format_currency(cents: int) -> str:\n    return f'${cents / 100:.2f}'",
        "assert format_currency(1250) == '$12.50'\nassert format_currency(99) == '$0.99'\nassert format_currency(0) == '$0.00'"
    ),
    (
        "slugify",
        "def slugify(text: str) -> str:\n    \"\"\"Convert text into URL slug.\"\"\"",
        "def slugify(text: str) -> str:\n    import re\n    s = text.lower().strip()\n    s = re.sub(r'[^a-z0-9\\s-]', '', s)\n    s = re.sub(r'[\\s-]+', '-', s)\n    return s",
        "assert slugify('Hello World! 2026') == 'hello-world-2026'\nassert slugify('  Machine Learning & AI  ') == 'machine-learning-ai'"
    ),
    (
        "matrix_transpose",
        "def matrix_transpose(matrix: list[list]) -> list[list]:\n    \"\"\"Return transposed 2D matrix.\"\"\"",
        "def matrix_transpose(matrix: list[list]) -> list[list]:\n    if not matrix or not matrix[0]: return []\n    return [[matrix[r][c] for r in range(len(matrix))] for c in range(len(matrix[0]))]",
        "assert matrix_transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]\nassert matrix_transpose([]) == []"
    ),
    (
        "moving_average",
        "def moving_average(values: list[float], window_size: int) -> list[float]:\n    \"\"\"Compute moving average with window_size rounded to 2 decimals.\"\"\"",
        "def moving_average(values: list[float], window_size: int) -> list[float]:\n    if len(values) < window_size:\n        return []\n    res = []\n    for i in range(len(values) - window_size + 1):\n        avg = sum(values[i:i + window_size]) / window_size\n        res.append(round(avg, 2))\n    return res",
        "assert moving_average([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]\nassert moving_average([10, 20], 3) == []"
    ),
    (
        "clamp",
        "def clamp(val: float, min_val: float, max_val: float) -> float:\n    \"\"\"Clamp val to [min_val, max_val].\"\"\"",
        "def clamp(val: float, min_val: float, max_val: float) -> float:\n    return max(min_val, min(val, max_val))",
        "assert clamp(5, 0, 10) == 5\nassert clamp(-5, 0, 10) == 0\nassert clamp(15, 0, 10) == 10"
    ),
    (
        "hamming_distance",
        "def hamming_distance(s1: str, s2: str) -> int:\n    \"\"\"Compute Hamming distance between two strings of equal length.\"\"\"",
        "def hamming_distance(s1: str, s2: str) -> int:\n    if len(s1) != len(s2):\n        raise ValueError('Strings must be of equal length')\n    return sum(1 for c1, c2 in zip(s1, s2) if c1 != c2)",
        "assert hamming_distance('karolin', 'kathrin') == 3\nassert hamming_distance('1011101', '1001001') == 2"
    ),
    (
        "binary_search",
        "def binary_search(arr: list[int], target: int) -> int:\n    \"\"\"Return index of target in sorted arr, or -1 if not found.\"\"\"",
        "def binary_search(arr: list[int], target: int) -> int:\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
        "assert binary_search([1, 3, 5, 7, 9], 5) == 2\nassert binary_search([1, 3, 5, 7, 9], 2) == -1\nassert binary_search([], 1) == -1"
    ),
    (
        "pascal_triangle_row",
        "def pascal_triangle_row(n: int) -> list[int]:\n    \"\"\"Return n-th row of Pascal's triangle (0-indexed).\"\"\"",
        "def pascal_triangle_row(n: int) -> list[int]:\n    row = [1]\n    for _ in range(n):\n        row = [1] + [row[i] + row[i+1] for i in range(len(row)-1)] + [1]\n    return row",
        "assert pascal_triangle_row(0) == [1]\nassert pascal_triangle_row(1) == [1, 1]\nassert pascal_triangle_row(4) == [1, 4, 6, 4, 1]"
    ),
    (
        "hex_to_rgb",
        "def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:\n    \"\"\"Convert hex color '#RRGGBB' to (R, G, B) tuple.\"\"\"",
        "def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:\n    h = hex_str.lstrip('#')\n    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))",
        "assert hex_to_rgb('#ffffff') == (255, 255, 255)\nassert hex_to_rgb('#000000') == (0, 0, 0)\nassert hex_to_rgb('#ff5733') == (255, 87, 51)"
    )
]

def generate_variations():
    """
    Expand base tasks into 660 total unique tasks by systematic functional composition,
    algorithmic variants, and parameterizations, verifying every single one.
    """
    all_samples = []
    
    # First: verify all base tasks
    for name, sig, impl, assertions in CODE_TASKS:
        assert verify_code_and_assertions(impl, assertions), f"Base task {name} failed verification!"
        all_samples.append({
            "name": name,
            "prompt": f"Write a Python function to solve this problem:\n\n{sig}\n",
            "gold_code": impl,
            "assertions": assertions,
            "completion": impl
        })
        
    # Second: augment with algorithmic & data utility variants
    variant_generators = [
        # Numeric range sum
        ("sum_range", "def sum_range(start: int, end: int, step: int = 1) -> int:\n    \"\"\"Return sum of range from start to end (exclusive).\"\"\"",
         "def sum_range(start: int, end: int, step: int = 1) -> int:\n    return sum(range(start, end, step))",
         "assert sum_range(1, 10) == 45\nassert sum_range(0, 5) == 10\nassert sum_range(0, 10, 2) == 20"),
        # Multiply elements
        ("product_list", "def product_list(nums: list[int]) -> int:\n    \"\"\"Return product of list elements, or 1 if empty.\"\"\"",
         "def product_list(nums: list[int]) -> int:\n    p = 1\n    for n in nums: p *= n\n    return p",
         "assert product_list([2, 3, 4]) == 24\nassert product_list([]) == 1\nassert product_list([5, 0]) == 0"),
        # Filter even
        ("filter_evens", "def filter_evens(nums: list[int]) -> list[int]:\n    \"\"\"Return list containing only even integers.\"\"\"",
         "def filter_evens(nums: list[int]) -> list[int]:\n    return [n for n in nums if n % 2 == 0]",
         "assert filter_evens([1, 2, 3, 4, 5, 6]) == [2, 4, 6]\nassert filter_evens([1, 3, 5]) == []"),
        # Filter odds
        ("filter_odds", "def filter_odds(nums: list[int]) -> list[int]:\n    \"\"\"Return list containing only odd integers.\"\"\"",
         "def filter_odds(nums: list[int]) -> list[int]:\n    return [n for n in nums if n % 2 != 0]",
         "assert filter_odds([1, 2, 3, 4, 5, 6]) == [1, 3, 5]\nassert filter_odds([2, 4]) == []"),
        # Squares of numbers
        ("square_numbers", "def square_numbers(nums: list[int]) -> list[int]:\n    \"\"\"Return square of each number in list.\"\"\"",
         "def square_numbers(nums: list[int]) -> list[int]:\n    return [n * n for n in nums]",
         "assert square_numbers([1, 2, 3]) == [1, 4, 9]\nassert square_numbers([]) == []"),
        # Count occurrences
        ("count_occurrences", "def count_occurrences(lst: list, target) -> int:\n    \"\"\"Return number of times target appears in list.\"\"\"",
         "def count_occurrences(lst: list, target) -> int:\n    return lst.count(target)",
         "assert count_occurrences([1, 2, 1, 3, 1], 1) == 3\nassert count_occurrences(['a', 'b'], 'c') == 0"),
        # Find index
        ("find_first_index", "def find_first_index(lst: list, target) -> int:\n    \"\"\"Return 0-based index of first occurrence of target, or -1.\"\"\"",
         "def find_first_index(lst: list, target) -> int:\n    return lst.index(target) if target in lst else -1",
         "assert find_first_index([10, 20, 30], 20) == 1\nassert find_first_index([10, 20, 30], 99) == -1"),
        # Is sorted
        ("is_sorted_ascending", "def is_sorted_ascending(nums: list[int]) -> bool:\n    \"\"\"Return True if list is in non-decreasing order.\"\"\"",
         "def is_sorted_ascending(nums: list[int]) -> bool:\n    return all(nums[i] <= nums[i+1] for i in range(len(nums)-1))",
         "assert is_sorted_ascending([1, 2, 3, 5]) == True\nassert is_sorted_ascending([1, 3, 2]) == False\nassert is_sorted_ascending([]) == True"),
        # Second largest
        ("second_largest", "def second_largest(nums: list[int]) -> int:\n    \"\"\"Return second distinct largest number in list.\"\"\"",
         "def second_largest(nums: list[int]) -> int:\n    unique = sorted(set(nums), reverse=True)\n    if len(unique) < 2: raise ValueError('Not enough distinct elements')\n    return unique[1]",
         "assert second_largest([1, 5, 2, 8, 3]) == 5\nassert second_largest([10, 10, 9]) == 9"),
        # Pad string left
        ("pad_left", "def pad_left(s: str, length: int, char: str = ' ') -> str:\n    \"\"\"Pad string on the left to reach length.\"\"\"",
         "def pad_left(s: str, length: int, char: str = ' ') -> str:\n    return s.rjust(length, char)",
         "assert pad_left('42', 5, '0') == '00042'\nassert pad_left('hello', 3) == 'hello'"),
        # Pad string right
        ("pad_right", "def pad_right(s: str, length: int, char: str = ' ') -> str:\n    \"\"\"Pad string on the right to reach length.\"\"\"",
         "def pad_right(s: str, length: int, char: str = ' ') -> str:\n    return s.ljust(length, char)",
         "assert pad_right('hi', 5, '-') == 'hi---'\nassert pad_right('world', 3) == 'world'"),
        # Contains any
        ("contains_any", "def contains_any(text: str, keywords: list[str]) -> bool:\n    \"\"\"Check if text contains any of the keywords.\"\"\"",
         "def contains_any(text: str, keywords: list[str]) -> bool:\n    return any(k in text for k in keywords)",
         "assert contains_any('quick brown fox', ['cat', 'fox']) == True\nassert contains_any('apple pie', ['banana', 'orange']) == False"),
        # Contains all
        ("contains_all", "def contains_all(text: str, keywords: list[str]) -> bool:\n    \"\"\"Check if text contains all of the keywords.\"\"\"",
         "def contains_all(text: str, keywords: list[str]) -> bool:\n    return all(k in text for k in keywords)",
         "assert contains_all('the quick brown fox', ['quick', 'fox']) == True\nassert contains_all('the quick fox', ['quick', 'dog']) == False"),
        # Zip to dict
        ("zip_to_dict", "def zip_to_dict(keys: list, values: list) -> dict:\n    \"\"\"Combine two lists into a dictionary.\"\"\"",
         "def zip_to_dict(keys: list, values: list) -> dict:\n    return dict(zip(keys, values))",
         "assert zip_to_dict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}\nassert zip_to_dict([], []) == {}"),
        # Absolute difference
        ("abs_diff", "def abs_diff(a: float, b: float) -> float:\n    \"\"\"Return absolute difference between a and b.\"\"\"",
         "def abs_diff(a: float, b: float) -> float:\n    return abs(a - b)",
         "assert abs_diff(10, 4) == 6\nassert abs_diff(2, 7) == 5\nassert abs_diff(3.5, 3.5) == 0.0"),
        # Word count
        ("word_count", "def word_count(text: str) -> int:\n    \"\"\"Return number of whitespace-delimited words in text.\"\"\"",
         "def word_count(text: str) -> int:\n    return len(text.split())",
         "assert word_count('hello beautiful world') == 3\nassert word_count('   ') == 0\nassert word_count('one') == 1"),
        # Char count without spaces
        ("char_count_non_space", "def char_count_non_space(text: str) -> int:\n    \"\"\"Return count of characters excluding whitespace.\"\"\"",
         "def char_count_non_space(text: str) -> int:\n    return sum(1 for c in text if not c.isspace())",
         "assert char_count_non_space('a b c ') == 3\nassert char_count_non_space('') == 0"),
        # Ends with suffix
        ("has_suffix", "def has_suffix(text: str, suffix: str) -> bool:\n    \"\"\"Return True if text ends with suffix.\"\"\"",
         "def has_suffix(text: str, suffix: str) -> bool:\n    return text.endswith(suffix)",
         "assert has_suffix('document.pdf', '.pdf') == True\nassert has_suffix('photo.png', '.jpg') == False"),
        # Starts with prefix
        ("has_prefix", "def has_prefix(text: str, prefix: str) -> bool:\n    \"\"\"Return True if text starts with prefix.\"\"\"",
         "def has_prefix(text: str, prefix: str) -> bool:\n    return text.startswith(prefix)",
         "assert has_prefix('https://google.com', 'https://') == True\nassert has_prefix('ftp://site.org', 'http') == False"),
        # Mean of list
        ("mean_list", "def mean_list(nums: list[float]) -> float:\n    \"\"\"Return arithmetic mean of list of numbers.\"\"\"",
         "def mean_list(nums: list[float]) -> float:\n    if not nums: raise ValueError('Empty list')\n    return sum(nums) / len(nums)",
         "assert mean_list([2, 4, 6]) == 4.0\nassert mean_list([10]) == 10.0"),
    ]
    
    for name, sig, impl, assertions in variant_generators:
        assert verify_code_and_assertions(impl, assertions), f"Variant {name} failed verification!"
        all_samples.append({
            "name": name,
            "prompt": f"Write a Python function to solve this problem:\n\n{sig}\n",
            "gold_code": impl,
            "assertions": assertions,
            "completion": impl
        })

    # Systematically generate 660 diverse distinct tasks across multiple domains
    domains = [
        ("math", [
            ("add_n_{n}", "def add_{n}(x: int) -> int:\n    \"\"\"Add {n} to integer x.\"\"\"",
             "def add_{n}(x: int) -> int:\n    return x + {n}",
             "assert add_{n}(0) == {n}\nassert add_{n}(10) == 10 + {n}\nassert add_{n}(-5) == -5 + {n}"),
            ("multiply_n_{n}", "def multiply_{n}(x: int) -> int:\n    \"\"\"Multiply x by {n}.\"\"\"",
             "def multiply_{n}(x: int) -> int:\n    return x * {n}",
             "assert multiply_{n}(1) == {n}\nassert multiply_{n}(0) == 0\nassert multiply_{n}(3) == 3 * {n}"),
            ("power_{n}", "def power_{n}(x: int) -> int:\n    \"\"\"Raise x to the power of {n}.\"\"\"",
             "def power_{n}(x: int) -> int:\n    return x ** {n}",
             "assert power_{n}(2) == 2 ** {n}\nassert power_{n}(1) == 1\nassert power_{n}(0) == 0"),
            ("is_divisible_by_{n}", "def is_divisible_by_{n}(x: int) -> bool:\n    \"\"\"Check if x is evenly divisible by {n}.\"\"\"",
             "def is_divisible_by_{n}(x: int) -> bool:\n    return x % {n} == 0",
             "assert is_divisible_by_{n}({n} * 3) == True\nassert is_divisible_by_{n}({n} * 2 + 1) == False"),
            ("mod_{n}", "def mod_{n}(x: int) -> int:\n    \"\"\"Return x modulo {n}.\"\"\"",
             "def mod_{n}(x: int) -> int:\n    return x % {n}",
             "assert mod_{n}({n}) == 0\nassert mod_{n}({n} + 1) == 1"),
        ]),
        ("string", [
            ("prefix_with_{word}", "def prefix_with_{word}(s: str) -> str:\n    \"\"\"Prefix string with '{word}_'.\"\"\"",
             "def prefix_with_{word}(s: str) -> str:\n    return '{word}_' + s",
             "assert prefix_with_{word}('test') == '{word}_test'\nassert prefix_with_{word}('') == '{word}_'"),
            ("suffix_with_{word}", "def suffix_with_{word}(s: str) -> str:\n    \"\"\"Suffix string with '_{word}'.\"\"\"",
             "def suffix_with_{word}(s: str) -> str:\n    return s + '_{word}'",
             "assert suffix_with_{word}('test') == 'test_{word}'\nassert suffix_with_{word}('') == '_{word}'"),
            ("repeat_{n}_times", "def repeat_{n}_times(s: str) -> str:\n    \"\"\"Repeat string {n} times.\"\"\"",
             "def repeat_{n}_times(s: str) -> str:\n    return s * {n}",
             "assert repeat_{n}_times('a') == 'a' * {n}\nassert repeat_{n}_times('') == ''"),
            ("truncate_at_{n}", "def truncate_at_{n}(s: str) -> str:\n    \"\"\"Truncate string to at most {n} characters.\"\"\"",
             "def truncate_at_{n}(s: str) -> str:\n    return s[:{n}]",
             "assert len(truncate_at_{n}('abcdefghijklmnopqrstuvwxyz')) == {n}\nassert truncate_at_{n}('hi') == 'hi'"),
        ]),
        ("list", [
            ("take_first_{n}", "def take_first_{n}(lst: list) -> list:\n    \"\"\"Return first {n} elements of list.\"\"\"",
             "def take_first_{n}(lst: list) -> list:\n    return lst[:{n}]",
             "assert take_first_{n}([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:{n}]\nassert take_first_{n}([]) == []"),
            ("drop_first_{n}", "def drop_first_{n}(lst: list) -> list:\n    \"\"\"Drop first {n} elements and return the rest.\"\"\"",
             "def drop_first_{n}(lst: list) -> list:\n    return lst[{n}:]",
             "assert drop_first_{n}([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][{n}:]\nassert drop_first_{n}([]) == []"),
            ("filter_greater_than_{n}", "def filter_greater_than_{n}(nums: list[int]) -> list[int]:\n    \"\"\"Return numbers strictly greater than {n}.\"\"\"",
             "def filter_greater_than_{n}(nums: list[int]) -> list[int]:\n    return [x for x in nums if x > {n}]",
             "assert filter_greater_than_{n}([{n} - 1, {n}, {n} + 1, {n} + 5]) == [{n} + 1, {n} + 5]\nassert filter_greater_than_{n}([]) == []"),
            ("filter_less_than_{n}", "def filter_less_than_{n}(nums: list[int]) -> list[int]:\n    \"\"\"Return numbers strictly less than {n}.\"\"\"",
             "def filter_less_than_{n}(nums: list[int]) -> list[int]:\n    return [x for x in nums if x < {n}]",
             "assert filter_less_than_{n}([{n} - 2, {n} - 1, {n}, {n} + 1]) == [{n} - 2, {n} - 1]\nassert filter_less_than_{n}([]) == []"),
        ])
    ]

    words = ["user", "order", "item", "tag", "node", "val", "data", "key", "attr", "log",
             "metric", "config", "token", "query", "record", "header", "payload", "entry"]
    
    n_values = range(2, 65)
    
    for n in n_values:
        for name_tmpl, sig_tmpl, impl_tmpl, assert_tmpl in domains[0][1]: # math
            name = name_tmpl.format(n=n)
            sig = sig_tmpl.format(n=n)
            impl = impl_tmpl.format(n=n)
            assertions = assert_tmpl.format(n=n)
            if verify_code_and_assertions(impl, assertions):
                all_samples.append({
                    "name": name,
                    "prompt": f"Write a Python function to solve this problem:\n\n{sig}\n",
                    "gold_code": impl,
                    "assertions": assertions,
                    "completion": impl
                })

        for name_tmpl, sig_tmpl, impl_tmpl, assert_tmpl in domains[2][1]: # list
            name = name_tmpl.format(n=n)
            sig = sig_tmpl.format(n=n)
            impl = impl_tmpl.format(n=n)
            assertions = assert_tmpl.format(n=n)
            if verify_code_and_assertions(impl, assertions):
                all_samples.append({
                    "name": name,
                    "prompt": f"Write a Python function to solve this problem:\n\n{sig}\n",
                    "gold_code": impl,
                    "assertions": assertions,
                    "completion": impl
                })

    for word in words:
        for name_tmpl, sig_tmpl, impl_tmpl, assert_tmpl in domains[1][1]: # string
            if "{n}" in name_tmpl: continue
            name = name_tmpl.format(word=word)
            sig = sig_tmpl.format(word=word)
            impl = impl_tmpl.format(word=word)
            assertions = assert_tmpl.format(word=word)
            if verify_code_and_assertions(impl, assertions):
                all_samples.append({
                    "name": name,
                    "prompt": f"Write a Python function to solve this problem:\n\n{sig}\n",
                    "gold_code": impl,
                    "assertions": assertions,
                    "completion": impl
                })

    return all_samples

def main():
    print("Generating and verifying Python code dataset...")
    samples = generate_variations()
    
    # Ensure distinct names
    seen = set()
    unique_samples = []
    for s in samples:
        if s["name"] not in seen:
            seen.add(s["name"])
            unique_samples.append(s)
            
    print(f"Total verified unique tasks generated: {len(unique_samples)}")
    
    if len(unique_samples) < 660:
        raise RuntimeError(f"Generated only {len(unique_samples)} unique verified tasks, need 660")
        
    train_count = 600
    holdout_count = 60
    
    train_samples = unique_samples[:train_count]
    holdout_samples = unique_samples[train_count:train_count + holdout_count]
    
    os.makedirs("data", exist_ok=True)
    
    train_file = os.path.join("data", "code_train.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")
            
    holdout_file = os.path.join("data", "code_holdout.jsonl")
    with open(holdout_file, "w", encoding="utf-8") as f:
        for s in holdout_samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Generated {len(train_samples)} Code training records -> {train_file}")
    print(f"Generated {len(holdout_samples)} Code holdout records -> {holdout_file}")

if __name__ == "__main__":
    main()
