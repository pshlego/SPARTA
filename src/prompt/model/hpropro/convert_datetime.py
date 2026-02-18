CONVERT_DATETIME="""Convert this string to a string in the format of "yy-mm-dd hh:mm:ss".
If there is no year, month or day, just use 2000/01/01;
If there is no minute or second, just use 00:00:00.

For example:
* input "2022/1/23 12:32:23", return "2022-01-23 12:32:23"
* input "23:12.232 (AM)", return "2000-01-01 00:23:12.232"
* input "12/22/2021", return "2021-12-22 00:00:00"
* input "January 12, 2021", return "2021-01-12 00:00:00"

This is the input:
[TIME]
Just give me the answer. Please return the result directly without any explanation."""
