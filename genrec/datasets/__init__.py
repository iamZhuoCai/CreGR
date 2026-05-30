from genrec.datasets.Disco.dataset import Disco

try:
    from genrec.datasets.AmazonReviews2023.dataset import AmazonReviews2023
except ImportError:
    AmazonReviews2023 = None
