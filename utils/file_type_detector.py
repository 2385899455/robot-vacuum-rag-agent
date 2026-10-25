"""知识库导入使用的物理文件类型检测。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

# 当前文档读取器支持的物理文件格式
# 使用 PhysicalFileType.TXT 得到枚举成员
class PhysicalFileType(str, Enum):
    """当前文档读取器支持的物理文件格式。"""

    TXT = "txt"
    PDF = "pdf"
    UNKNOWN = "unknown"

# 文档类型检测器
# 根据文件路径判断文件属于哪种物理类型
class PhysicalFileTypeDetector:
    """
    识别策略：
    1. 优先检查文件头；
    2. 文件头无法判断时，再检查扩展名；
    3. PDF 有稳定的文件头；
    4. TXT 没有统一的文件头，所以主要依赖扩展名。
    """

    # PDF有稳定的文件头，TXT文件没有统一文件头，主要依赖扩展名，这里的命名为二进制
    _PDF_SIGNATURE = b"%PDF-"

    @classmethod
    def detect(cls, path: str | Path) -> PhysicalFileType:
        # 不直接使用suffix判断是防止PDF被修改扩展名，扩展名错误

        # 先获取文件路径，将传入路径转换成Path对象
        file_path = Path(path)

        if not file_path.is_file():
            return PhysicalFileType.UNKNOWN

        try:
            # rb 按照二进制读取
            with file_path.open("rb") as file_obj:
                # 只读取开头前八个字节并保存到header中，因为只需要获取到 %PDF- 这个签名
                header = file_obj.read(8)
        except OSError:
            return PhysicalFileType.UNKNOWN

        # 因为文件是二进制读取的，需要用字节串进行比较是否是PDF文件，不能直接拿二进制内容和普通字符串比较
        # startswith 判断一个字符串或字节串是否以指定内容开头
        if header.startswith(cls._PDF_SIGNATURE):
            return PhysicalFileType.PDF

        # 如果不是以 %PDF- 开头的文件
        # 获取文件扩展名，比如 "demo.PDF" 得到 .pdf
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return PhysicalFileType.TXT
        if suffix == ".pdf":
            return PhysicalFileType.PDF

        return PhysicalFileType.UNKNOWN
