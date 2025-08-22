import os
import time
import logging
import re
import json
from tabulate import tabulate
from pathlib import Path
from typing import Iterable, List

from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, FormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

_log = logging.getLogger(__name__)

def _process_chunk(pdf_paths, pdf_backend, output_dir, num_threads, metadata_lookup, debug_data_path):
    """Helper function to process a chunk of PDFs in a separate process."""
    # Create a new parser instance for this process
    parser = PDFParser(
        pdf_backend=pdf_backend,
        output_dir=output_dir,
        num_threads=num_threads,
        csv_metadata_path=None  # Metadata lookup is passed directly
    )
    parser.metadata_lookup = metadata_lookup
    parser.debug_data_path = debug_data_path
    parser.parse_and_export(pdf_paths)
    return f"Processed {len(pdf_paths)} PDFs."

class PDFParser:
    def __init__(
        self,
        pdf_backend=DoclingParseV2DocumentBackend,
        output_dir: Path = Path("./parsed_pdfs"),
        num_threads: int = None,
        csv_metadata_path: Path = None,
    ):
        self.pdf_backend = pdf_backend
        self.output_dir = output_dir
        self.doc_converter = self._create_document_converter()
        self.num_threads = num_threads
        self.metadata_lookup = {}
        self.debug_data_path = None

        if csv_metadata_path is not None:
            self.metadata_lookup = self._parse_csv_metadata(csv_metadata_path)

        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

    @staticmethod
    def _parse_csv_metadata(csv_path: Path) -> dict:
        """Parse CSV file and create a lookup dictionary with sha1 as key."""
        import csv
        metadata_lookup = {}

        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Handle both old and new CSV formats for company name
                company_name = row.get('company_name', row.get('name', '')).strip('"')
                metadata_lookup[row['sha1']] = {
                    'company_name': company_name
                }
        return metadata_lookup

    def _create_document_converter(self) -> "DocumentConverter": # type: ignore
        """Creates and returns a DocumentConverter with default pipeline options."""
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

        pipeline_options = PdfPipelineOptions(artifacts_path=Path("./docling_models"))
        pipeline_options.do_ocr = True
        ocr_options = EasyOcrOptions(lang=['ch_sim'], force_full_page_ocr=False, download_enabled=False, model_storage_directory=str(Path("./docling_models/EasyOcr")))
        pipeline_options.ocr_options = ocr_options
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=self.pdf_backend
            )
        }

        return DocumentConverter(format_options=format_options)

    def convert_documents(self, input_doc_paths: List[Path]) -> Iterable[ConversionResult]:
        conv_results = self.doc_converter.convert_all(source=input_doc_paths)
        return conv_results

    def process_documents(self, conv_results: Iterable[ConversionResult]):
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        failure_count = 0

        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                processor = JsonReportProcessor(metadata_lookup=self.metadata_lookup, debug_data_path=self.debug_data_path)

                # Normalize the document data to ensure sequential pages
                data = conv_res.document.export_to_dict()
                normalized_data = self._normalize_page_sequence(data)

                processed_report = processor.assemble_report(conv_res, normalized_data)
                doc_filename = conv_res.input.file.stem
                if self.output_dir is not None:
                    with (self.output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
                        json.dump(processed_report, fp, indent=2, ensure_ascii=False)
            else:
                failure_count += 1
                _log.info(f"Document {conv_res.input.file} failed to convert.")

        _log.info(f"Processed {success_count + failure_count} docs, of which {failure_count} failed")
        return success_count, failure_count

    def _normalize_page_sequence(self, data: dict) -> dict:
        """Ensure that page numbers in content are sequential by filling gaps with empty pages."""
        if 'content' not in data:
            return data

        # Create a copy of the data to modify
        normalized_data = data.copy()

        # Get existing page numbers and find max page
        existing_pages = {page['page'] for page in data['content']}
        max_page = max(existing_pages)

        # Create template for empty page
        empty_page_template = {
            "content": [],
            "page_dimensions": {}  # or some default dimensions if needed
        }

        # Create new content array with all pages
        new_content = []
        for page_num in range(1, max_page + 1):
            # Find existing page or create empty one
            page_content = next(
                (page for page in data['content'] if page['page'] == page_num),
                {"page": page_num, **empty_page_template}
            )
            new_content.append(page_content)

        normalized_data['content'] = new_content
        return normalized_data

    def parse_and_export(self, input_doc_paths: List[Path] = None, doc_dir: Path = None):
        start_time = time.time()
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_docs = len(input_doc_paths)
        _log.info(f"Starting to process {total_docs} documents")

        conv_results = self.convert_documents(input_doc_paths)
        success_count, failure_count = self.process_documents(conv_results=conv_results)
        elapsed_time = time.time() - start_time

        if failure_count > 0:
            error_message = f"Failed converting {failure_count} out of {total_docs} documents."
            failed_docs = "Paths of failed docs:\n" + '\n'.join(str(path) for path in input_doc_paths)
            _log.error(error_message)
            _log.error(failed_docs)
            raise RuntimeError(error_message)

        _log.info(f"{'#'*50}\nCompleted in {elapsed_time:.2f} seconds. Successfully converted {success_count}/{total_docs} documents.\n{'#'*50}")

    def parse_and_export_parallel(
        self,
        input_doc_paths: List[Path] = None,
        doc_dir: Path = None,
        optimal_workers: int = 10,
        chunk_size: int = None
    ):
        """Parse PDF files in parallel using multiple processes.

        Args:
            input_doc_paths: List of paths to PDF files to process
            doc_dir: Directory containing PDF files (used if input_doc_paths is None)
            optimal_workers: Number of worker processes to use. If None, uses CPU count.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Get input paths if not provided
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_pdfs = len(input_doc_paths)
        _log.info(f"Starting parallel processing of {total_pdfs} documents")

        cpu_count = multiprocessing.cpu_count()

        # Calculate optimal workers if not specified
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)

        if chunk_size is None:
            # Calculate chunk size (ensure at least 1)
            chunk_size = max(1, total_pdfs // optimal_workers)

        # Split documents into chunks
        chunks = [
            input_doc_paths[i : i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]

        start_time = time.time()
        processed_count = 0

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Schedule all tasks
            futures = [
                executor.submit(
                    _process_chunk,
                    chunk,
                    self.pdf_backend,
                    self.output_dir,
                    self.num_threads,
                    self.metadata_lookup,
                    self.debug_data_path
                )
                for chunk in chunks
            ]

            # Wait for completion and log results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed_count += int(result.split()[1])  # Extract number from "Processed X PDFs"
                    _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#'*50}")
                except Exception as e:
                    _log.error(f"Error processing chunk: {str(e)}")
                    raise

        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


class JsonReportProcessor:
    def __init__(self, metadata_lookup: dict = None, debug_data_path: Path = None):
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path

    def assemble_report(self, conv_result, normalized_data=None):
        """Assemble the report using either normalized data or raw conversion result."""
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dict()
        assembled_report = {}
        assembled_report['metainfo'] = self.assemble_metainfo(data)
        assembled_report['content'] = self.assemble_content(data)
        assembled_report['tables'] = self.assemble_tables(conv_result.document.tables, data, conv_result)
        assembled_report['pictures'] = self.assemble_pictures(data)
        self.debug_data(data)
        return assembled_report

    def assemble_metainfo(self, data):
        metainfo = {}
        sha1_name = data['origin']['filename'].rsplit('.', 1)[0]
        metainfo['sha1_name'] = sha1_name
        metainfo['pages_amount'] = len(data.get('pages', []))
        metainfo['text_blocks_amount'] = len(data.get('texts', []))
        metainfo['tables_amount'] = len(data.get('tables', []))
        metainfo['pictures_amount'] = len(data.get('pictures', []))
        metainfo['equations_amount'] = len(data.get('equations', []))
        metainfo['footnotes_amount'] = len([t for t in data.get('texts', []) if t.get('label') == 'footnote'])

        # Add CSV metadata if available
        if self.metadata_lookup and sha1_name in self.metadata_lookup:
            csv_meta = self.metadata_lookup[sha1_name]
            metainfo['company_name'] = csv_meta['company_name']

        return metainfo

    def process_table(self, table_data):
        # Implement your table processing logic here
        return 'processed_table_content'

    def debug_data(self, data):
        if self.debug_data_path is None:
            return
        doc_name = data['name']
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def expand_groups(self, body_children, groups):
        expanded_children = []

        for item in body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'groups':
                    group = groups[ref_num]
                    group_id = ref_num
                    group_name = group.get('name', '')
                    group_label = group.get('label', '')

                    for child in group['children']:
                        child_copy = child.copy()
                        child_copy['group_id'] = group_id
                        child_copy['group_name'] = group_name
                        child_copy['group_label'] = group_label
                        expanded_children.append(child_copy)
                else:
                    expanded_children.append(item)
            else:
                expanded_children.append(item)

        return expanded_children

    def _process_text_reference(self, ref_num, data):
        """Helper method to process text references and create content items.

        Args:
            ref_num (int): Reference number for the text item
            data (dict): Document data dictionary

        Returns:
            dict: Processed content item with text information
        """
        text_item = data['texts'][ref_num]
        item_type = text_item['label']
        content_item = {
            'text': text_item.get('text', ''),
            'type': item_type,
            'text_id': ref_num
        }

        # Add 'orig' field only if it differs from 'text'
        orig_content = text_item.get('orig', '')
        if orig_content != text_item.get('text', ''):
            content_item['orig'] = orig_content

        # Add additional fields if they exist
        if 'enumerated' in text_item:
            content_item['enumerated'] = text_item['enumerated']
        if 'marker' in text_item:
            content_item['marker'] = text_item['marker']

        return content_item

    def assemble_content(self, data):
        pages = {}
        # Expand body children to include group references
        body_children = data['body']['children']
        groups = data.get('groups', [])
        expanded_body_children = self.expand_groups(body_children, groups)

        # Process body content
        for item in expanded_body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'texts':
                    text_item = data['texts'][ref_num]
                    content_item = self._process_text_reference(ref_num, data)

                    # Add group information if available
                    if 'group_id' in item:
                        content_item['group_id'] = item['group_id']
                        content_item['group_name'] = item['group_name']
                        content_item['group_label'] = item['group_label']

                    # Get page number from prov
                    if 'prov' in text_item and text_item['prov']:
                        page_num = text_item['prov'][0]['page_no']

                        # Initialize page if not exists
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': text_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

                elif ref_type == 'tables':
                    table_item = data['tables'][ref_num]
                    content_item = {
                        'type': 'table',
                        'table_id': ref_num
                    }

                    if 'prov' in table_item and table_item['prov']:
                        page_num = table_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': table_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

                elif ref_type == 'pictures':
                    picture_item = data['pictures'][ref_num]
                    content_item = {
                        'type': 'picture',
                        'picture_id': ref_num
                    }

                    if 'prov' in picture_item and picture_item['prov']:
                        page_num = picture_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': picture_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def assemble_tables(self, tables, data, conv_result):
        assembled_tables = []
        for i, table in enumerate(tables):
            table_json_obj = table.model_dump()
            table_md = self._table_to_md(table_json_obj)
            table_html = table.export_to_html(doc=conv_result.document)

            table_data = data['tables'][i]
            table_page_num = table_data['prov'][0]['page_no']
            table_bbox = table_data['prov'][0]['bbox']
            table_bbox = [
                table_bbox['l'],
                table_bbox['t'],
                table_bbox['r'],
                table_bbox['b']
            ]

            # Get rows and columns from the table data structure
            nrows = table_data['data']['num_rows']
            ncols = table_data['data']['num_cols']

            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            table_obj = {
                'table_id': ref_num,
                'page': table_page_num,
                'bbox': table_bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'json': table_json_obj
            }
            assembled_tables.append(table_obj)
        return assembled_tables

    def _table_to_md(self, table):
        # Extract text from grid cells
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)

        # Check if the table has headers
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                md_table = tabulate(
                    table_data[1:], headers=table_data[0], tablefmt="github"
                )
            except ValueError:
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        else:
            md_table = tabulate(table_data, tablefmt="github")

        return md_table

    def assemble_pictures(self, data):
        assembled_pictures = []
        for i, picture in enumerate(data['pictures']):
            children_list = self._process_picture_block(picture, data)

            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            picture_page_num = picture['prov'][0]['page_no']
            picture_bbox = picture['prov'][0]['bbox']
            picture_bbox = [
                picture_bbox['l'],
                picture_bbox['t'],
                picture_bbox['r'],
                picture_bbox['b']
            ]

            picture_obj = {
                'picture_id': ref_num,
                'page': picture_page_num,
                'bbox': picture_bbox,
                'children': children_list,
            }
            assembled_pictures.append(picture_obj)
        return assembled_pictures

    def _process_picture_block(self, picture, data):
        children_list = []

        for item in picture['children']:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'texts':
                    content_item = self._process_text_reference(ref_num, data)

                    children_list.append(content_item)

        return children_list

class PageTextPreparation:
    """
    Cleans and formats page blocks according to rules, handling consecutive
    groups for tables, lists, and footnotes.
    """

    def __init__(self, use_serialized_tables: bool = False, serialized_tables_instead_of_markdown: bool = False):
        """Initialize with option to add serialized tables to markdown ones."""
        self.use_serialized_tables = use_serialized_tables
        self.serialized_tables_instead_of_markdown = serialized_tables_instead_of_markdown
        self.report_data = None # Initialize report_data

    def process_reports(
        self,
        reports_dir: Path = None,
        reports_paths: List[Path] = None,
        output_dir: Path = None
    ):
        """
        Process reports from a directory or list of paths, returning a list of processed reports
        and saving them to an output directory if specified.
        """
        all_reports = []

        if reports_dir:
            reports_paths = list(reports_dir.glob('*.json'))

        for report_path in reports_paths:
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)

            full_report_text = self.process_report(report_data)
            report = {"metainfo": report_data['metainfo'], "content": full_report_text}
            all_reports.append(report)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                    json.dump(report, file, indent=2, ensure_ascii=False)

        return all_reports

    def process_report(self, report_data):
        """
        Process a single report, returning a list of processed pages and printing a message if corrections were made.
        """
        self.report_data = report_data
        processed_pages = []
        total_corrections = 0
        corrections_list = []

        for page_content in self.report_data["content"]:
            page_number = page_content["page"]
            page_text = self.prepare_page_text(page_number)
            cleaned_text, corrections_count, corrections = self._clean_text(page_text)
            total_corrections += corrections_count
            corrections_list.extend(corrections)
            page_data = {
                "page": page_number,
                "text": cleaned_text
            }
            processed_pages.append(page_data)

        if total_corrections > 0:
            print(
                f"Fixed {total_corrections} occurrences in the file "
                f"{self.report_data['metainfo']['sha1_name']}"
            )
            print(corrections_list[:30])

        processed_report = {
            "chunks": None,
            "pages": processed_pages
        }

        return processed_report

    def prepare_page_text(self, page_number):
        """Main method to process page blocks and return assembled string."""
        page_data = self._get_page_data(page_number)
        if not page_data or "content" not in page_data:
            return ""

        blocks = page_data["content"]

        filtered_blocks = self._filter_blocks(blocks)
        final_blocks = self._apply_formatting_rules(filtered_blocks)

        if final_blocks:
            final_blocks[0] = final_blocks[0].lstrip()
            final_blocks[-1] = final_blocks[-1].rstrip()

        return "\n".join(final_blocks)

    def _get_page_data(self, page_number):
        """Returns page dict for given page number, or None if not found."""
        all_pages = self.report_data.get("content", [])
        for page in all_pages:
            if page.get("page") == page_number:
                return page
        return None

    def _filter_blocks(self, blocks):
        """Remove blocks of ignored types."""
        ignored_types = {"page_footer", "picture"}
        filtered_blocks = []
        for block in blocks:
            block_type = block.get("type")
            if block_type in ignored_types:
                continue
            filtered_blocks.append(block)
        return filtered_blocks

    def _clean_text(self, text):
        """Clean text using regex substitutions and count corrections."""
        command_mapping = {
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'period': '.',
            'comma': ',',
            'colon': ":",
            'hyphen': "-",
            'percent': '%',
            'dollar': '$',
            'space': ' ',
            'plus': '+',
            'minus': '-',
            'slash': '/',
            'asterisk': '*',
            'lparen': '(',
            'rparen': ')',
            'parenright': ')',
            'parenleft': '(',
            'wedge.1_E': '',
        }

        recognized_commands = "|".join(command_mapping.keys())
        slash_command_pattern = rf"/({recognized_commands})(\.pl\.tnum|\.tnum\.pl|\.pl|\.tnum|\.case|\.sups)"

        occurrences_amount = len(re.findall(slash_command_pattern, text))
        occurrences_amount += len(re.findall(r'glyph<[^>]*>', text))
        occurrences_amount += len(re.findall(r'/([A-Z])\.cap', text))

        corrections = []

        def replace_command(match):
            base_command = match.group(1)
            replacement = command_mapping.get(base_command)
            if replacement is not None:
                corrections.append((match.group(0), replacement))
            return replacement if replacement is not None else match.group(0)

        def replace_glyph(match):
            corrections.append((match.group(0), ''))
            return ''

        def replace_cap(match):
            original = match.group(0)
            replacement = match.group(1)
            corrections.append((original, replacement))
            return replacement

        text = re.sub(slash_command_pattern, replace_command, text)
        text = re.sub(r'glyph<[^>]*>', replace_glyph, text)
        text = re.sub(r'/([A-Z])\.cap', replace_cap, text)

        return text, occurrences_amount, corrections

    def _block_ends_with_colon(self, block):
        """Check if block text ends with colon for relevant block types."""
        block_type = block.get("type")
        text = block.get("text", "").rstrip()
        if block_type in {"text", "caption", "section_header", "paragraph"}:
            return text.endswith(":")
        return False

    def _apply_formatting_rules(self, blocks):
        """Transform blocks according to formatting rules."""
        page_header_in_first_3 = False
        section_header_in_first_3 = False
        for blk in blocks[:3]:
            if blk["type"] == "page_header":
                page_header_in_first_3 = True
            if blk["type"] == "section_header":
                section_header_in_first_3 = True

        final_blocks = []
        first_section_header_index = 0

        i = 0
        n = len(blocks)

        while i < n:
            block = blocks[i]
            block_type = block.get("type")
            text = block.get("text", "").strip()

            # Handle headers
            if block_type == "page_header":
                prefix = "\n# " if i < 3 else "\n## "
                final_blocks.append(f"{prefix}{text}\n")
                i += 1
                continue

            if block_type == "section_header":
                first_section_header_index += 1
                if (
                    first_section_header_index == 1
                    and i < 3
                    and not page_header_in_first_3
                ):
                    prefix = "\n# "
                else:
                    prefix = "\n## "
                final_blocks.append(f"{prefix}{text}\n")
                i += 1
                continue

            if block_type == "paragraph":
                if self._block_ends_with_colon(block) and i + 1 < n:
                    next_block_type = blocks[i + 1].get("type")
                    if next_block_type not in ("table", "list_item"):
                        final_blocks.append(f"\n### {text}\n")
                        i += 1
                        continue
                else:
                    final_blocks.append(f"\n### {text}\n")
                    i += 1
                    continue

            # Handle table groups
            if block_type == "table" or (
                self._block_ends_with_colon(block)
                and i + 1 < n
                and blocks[i + 1].get("type") == "table"
            ):
                group_blocks = []
                header_for_table = None
                if self._block_ends_with_colon(block) and i + 1 < n:
                    header_for_table = block
                    table_block = blocks[i + 1]
                    i += 2
                else:
                    table_block = block
                    i += 1

                if header_for_table:
                    group_blocks.append(header_for_table)
                group_blocks.append(table_block)

                footnote_candidates_start = i
                if i < n:
                    maybe_text_block = blocks[i]
                    if maybe_text_block.get("type") == "text":
                        if (i + 1 < n) and (blocks[i + 1].get("type") == "footnote"):
                            group_blocks.append(maybe_text_block)
                            i += 1

                while i < n and blocks[i].get("type") == "footnote":
                    group_blocks.append(blocks[i])
                    i += 1

                group_text = self._render_table_group(group_blocks)
                final_blocks.append(group_text)
                continue

            # Handle list groups
            if block_type == "list_item" or (
                self._block_ends_with_colon(block)
                and i + 1 < n
                and blocks[i + 1].get("type") == "list_item"
            ):
                group_blocks = []
                if self._block_ends_with_colon(block) and i + 1 < n:
                    header_for_list = block
                    i += 1
                    group_blocks.append(header_for_list)

                while i < n and blocks[i].get("type") == "list_item":
                    group_blocks.append(blocks[i])
                    i += 1

                if i < n and blocks[i].get("type") == "text":
                    if (i + 1 < n) and (blocks[i + 1].get("type") == "footnote"):
                        group_blocks.append(blocks[i])
                        i += 1

                while i < n and blocks[i].get("type") == "footnote":
                    group_blocks.append(blocks[i])
                    i += 1

                group_text = self._render_list_group(group_blocks)
                final_blocks.append(group_text)
                continue

            # Handle normal blocks
            if block_type in (
                "text",
                "caption",
                "footnote",
                "checkbox_selected",
                "checkbox_unselected",
                "formula",
            ):
                if not text.strip():
                    i += 1
                    continue
                else:
                    final_blocks.append(f"{text}\n")
                    i += 1
                continue

            raise ValueError(f"Unknown block type: {block_type}")

        return final_blocks

    def _render_table_group(self, group_blocks):
        """Render table group with optional header, text and footnotes."""
        chunk = []
        for blk in group_blocks:
            blk_type = blk.get("type")
            blk_text = blk.get("text", "").strip()
            if blk_type in {"text", "caption", "section_header", "paragraph"}:
                chunk.append(f"{blk_text}\n")

            elif blk_type == "table":
                table_id = blk.get("table_id")
                if table_id is None:
                    continue
                table_markdown = self._get_table_by_id(table_id)
                chunk.append(f"{table_markdown}\n")

            elif blk_type == "footnote":
                chunk.append(f"{blk_text}\n")

            elif blk_type == "text":
                chunk.append(f"{blk_text}\n")

            else:
                raise ValueError(f"Unexpected block type in table group: {blk_type}")

        return "\n" + "".join(chunk) + "\n"

    def _render_list_group(self, group_blocks):
        """Render list group with optional header, text and footnotes."""
        chunk = []
        for blk in group_blocks:
            blk_type = blk.get("type")
            blk_text = blk.get("text", "").strip()
            if blk_type in {"text", "caption", "section_header", "paragraph"}:
                chunk.append(f"{blk_text}\n")

            elif blk_type == "list_item":
                chunk.append(f"- {blk_text}\n")

            elif blk_type == "footnote":
                chunk.append(f"{blk_text}\n")

            elif blk_type == "checkbox_selected":
                chunk.append(f"[x] {blk_text}\n")

            elif blk_type == "checkbox_unselected":
                chunk.append(f"[ ] {blk_text}\n")

            else:
                chunk.append(f"{blk_text}\n")

        return "\n" + "".join(chunk) + "\n"

    def _get_table_by_id(self, table_id):
        """Get table representation by ID from report data.
        Returns markdown or serialized text based on configuration."""
        for t in self.report_data.get("tables", []):
            if t.get("table_id") == table_id:
                if self.use_serialized_tables:
                    return self._get_serialized_table_text(t, self.serialized_tables_instead_of_markdown)
                # Call the static method _table_to_md for markdown conversion
                return self._table_to_md(t.get("json", {})) # Pass the json data for conversion
        raise ValueError(f"Table with ID={table_id} not found in report_data!")

    def _get_serialized_table_text(self, table, serialized_tables_instead_of_markdown):
        """Convert serialized table format to text string.

        Args:
            table: Table object containing serialized data

        Returns:
            String containing concatenated information blocks or markdown as fallback
        """
        if not table.get("serialized"):
            return self._table_to_md(table.get("json", {})) # Fallback to markdown if no serialized data

        info_blocks = table["serialized"].get("information_blocks", [])
        text_blocks = [block["information_block"] for block in info_blocks]
        serialized_text = "\n".join(text_blocks)
        if serialized_tables_instead_of_markdown:
            return serialized_text
        else:
            markdown = self._table_to_md(table.get("json", {})) # Get markdown
            combined_text = f"{markdown}\nDescription of the table entities:\n{serialized_text}"
            return combined_text

    @staticmethod
    def _table_to_md(table):
        # Extract text from grid cells
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)

        # Check if the table has headers
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                md_table = tabulate(
                    table_data[1:], headers=table_data[0], tablefmt="github"
                )
            except ValueError:
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        else:
            md_table = tabulate(table_data, tablefmt="github")

        return md_table

    def export_to_markdown(self, reports_dir: Path, output_dir: Path):
        """Export processed reports to markdown files.

        Args:
            reports_dir: Directory containing JSON report files
            output_dir: Directory where markdown files will be saved
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in reports_dir.glob("*.json"):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            processed_report = self.process_report(report_data)

            document_text = ""
            for page in processed_report['pages']:
                document_text += f"\n\n---\n\n# Page {page['page']}\n\n"
                document_text += page['text']

            report_name = report_data['metainfo']['sha1_name']
            with open(output_dir / f"{report_name}.md", "w", encoding="utf-8") as f:
                f.write(document_text)
