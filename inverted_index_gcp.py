
from collections import Counter, defaultdict
import itertools
from pathlib import Path
import pickle
from google.cloud import storage
from contextlib import closing


PROJECT_ID = 'uni-project-480107'


def get_bucket(bucket_name):
    return storage.Client(project=PROJECT_ID).bucket(bucket_name)


def _open(path, mode, bucket=None):
    if bucket is None:
        return open(path, mode)
    return bucket.blob(path).open(mode)


BLOCK_SIZE = 1999998
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1


class MultiFileWriter:
    def __init__(self, base_dir, name, bucket_name=None):
        self._base_dir = str(base_dir)  # Store as string
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)


        def get_path(i):
            filename = f'{name}_{i:03}.bin'
            if bucket_name:
                return f"{self._base_dir}/{filename}"
            else:
                return str(Path(self._base_dir) / filename)

        self._file_gen = (_open(get_path(i), 'wb', self._bucket) for i in itertools.count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            if remaining == 0:
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])


            if hasattr(self._f, 'name'):
                name = Path(self._f.name).name
            else:

                name = self._f._blob.name.split('/')[-1]

            locs.append((name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()


class MultiFileReader:
    def __init__(self, base_dir, bucket_name=None):
        self._base_dir = str(base_dir)  # Ensure string
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if self._bucket:

                if f_name.startswith(f"{self._base_dir}/"):
                    full_path = f_name
                else:
                    full_path = f"{self._base_dir}/{f_name}"
            else:
                # Local Windows logic
                full_path = str(Path(self._base_dir) / f_name)

            if full_path not in self._open_files:
                self._open_files[full_path] = _open(full_path, 'rb', self._bucket)
            f = self._open_files[full_path]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class InvertedIndex:
    def __init__(self, docs={}):
        self.total_corpus_terms = 0
        self.N = 0
        self.doc_len = {}
        self.document_frequencey_per_term = Counter()
        self.df = self.document_frequencey_per_term
        self.term_total = Counter()
        self._posting_list = defaultdict(list)
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        self.total_corpus_terms += len(tokens)
        self.N += 1
        self.doc_len[doc_id] = len(tokens)
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name, bucket_name=None):
        self._write_globals(base_dir, name, bucket_name)

    def _write_globals(self, base_dir, name, bucket_name):
        # FIX: Force forward slash for GCS
        if bucket_name:
            path = f"{base_dir}/{name}.pkl"
        else:
            path = str(Path(base_dir) / f'{name}.pkl')

        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'wb', bucket) as f:
            pickle.dump(self, f)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self, base_dir, bucket_name=None):
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    def read_a_posting_list(self, base_dir, w, bucket_name=None):
        posting_list = []
        if not w in self.posting_locs:
            return posting_list
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * TUPLE_SIZE)
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return posting_list

    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name=None):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl:
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                locs = writer.write(b)
                posting_locs[w].extend(locs)


            if bucket_name:
                path = f"{base_dir}/{bucket_id}_posting_locs.pickle"
            else:
                path = str(Path(base_dir) / f'{bucket_id}_posting_locs.pickle')

            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, 'wb', bucket) as f:
                pickle.dump(posting_locs, f)
        return bucket_id

    @staticmethod
    def read_index(base_dir, name, bucket_name=None):

        if bucket_name:
            path = f"{base_dir}/{name}.pkl"
        else:
            path = str(Path(base_dir) / f'{name}.pkl')

        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'rb', bucket) as f:
            index = pickle.load(f)


            if not hasattr(index, 'df'):
                index.df = index.document_frequencey_per_term

            return index