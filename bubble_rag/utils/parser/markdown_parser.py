import re

class RAGFlowMarkdownParser:
    def __init__(self, chunk_token_num=128):
        self.chunk_token_num = int(chunk_token_num)

    def extract_tables_and_remainder(self, markdown_text):
        tables = []
        remainder = markdown_text
        if "|" in markdown_text: # for optimize performance
            # Standard Markdown table
            border_table_pattern = re.compile(
                r'''
                (?:\n|^)                     
                (?:\|.*?\|.*?\|.*?\n)        
                (?:\|(?:\s*[:-]+[-| :]*\s*)\|.*?\n) 
                (?:\|.*?\|.*?\|.*?\n)+
            ''', re.VERBOSE)
            border_tables = border_table_pattern.findall(markdown_text)
            tables.extend(border_tables)
            remainder = border_table_pattern.sub('', remainder)

            # Borderless Markdown table
            no_border_table_pattern = re.compile(
                r'''
                (?:\n|^)                 
                (?:\S.*?\|.*?\n)
                (?:(?:\s*[:-]+[-| :]*\s*).*?\n)
                (?:\S.*?\|.*?\n)+
                ''', re.VERBOSE)
            no_border_tables = no_border_table_pattern.findall(remainder)
            tables.extend(no_border_tables)
            remainder = no_border_table_pattern.sub('', remainder)

        if "<table>" in remainder.lower(): # for optimize performance
            #HTML table extraction - handle possible html/body wrapper tags
            html_table_pattern = re.compile(
            r'''
            (?:\n|^)
            \s*
            (?:
                # case1: <html><body><table>...</table></body></html>
                (?:<html[^>]*>\s*<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>\s*</html>)
                |
                # case2: <body><table>...</table></body>
                (?:<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>)
                |
                # case3: only<table>...</table>
                (?:<table[^>]*>.*?</table>)
            )
            \s*
            (?=\n|$)
            ''',
            re.VERBOSE | re.DOTALL | re.IGNORECASE
            )
            html_tables = html_table_pattern.findall(remainder)
            tables.extend(html_tables)
            remainder = html_table_pattern.sub('', remainder)

        return remainder, tables
