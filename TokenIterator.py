import io
import json

class TokenIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line) + 1
                full_line = line[:-1].decode("utf-8")
                line_data = json.loads(full_line.lstrip("data:").rstrip("/n"))
                return line_data["token"]["text"]
            chunk = next(self.byte_iterator)
            self.buffer.seek(0, io.SEEK_END)
            print(chunk)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])
