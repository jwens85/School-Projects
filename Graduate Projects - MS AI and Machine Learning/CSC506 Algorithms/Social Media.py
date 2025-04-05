class Post:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

class HashTable:
    def __init__(self, size=100):
        self.size = size
        self.buckets = [None] * size

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash(key)
        new_post = Post(key, value)

        if self.buckets[index] is None:
            self.buckets[index] = new_post
        else:
            current = self.buckets[index]
            while current:
                if current.key == key:
                    current.value = value
                    return
                if current.next is None:
                    break
                current = current.next
            current.next = new_post

    def get(self, key):
        index = self._hash(key)
        current = self.buckets[index]

        while current:
            if current.key == key:
                return current.value
            current = current.next
        return None

class Recommender:
    def __init__(self):
        self.user_ratings = HashTable()

    def add_ratings(self, user_id, ratings):
        self.user_ratings.insert(user_id, ratings)

    def recommend(self, target_user):
        target_ratings = self.user_ratings.get(target_user)
        if target_ratings is None:
            return "User not found or no ratings available."

        total_scores = {}
        rating_counts = {}

        for bucket in self.user_ratings.buckets:
            current = bucket
            while current:
                if current.key != target_user:
                    other_ratings = current.value
                    for item, rating in other_ratings.items():
                        if item not in total_scores:
                            total_scores[item] = 0
                            rating_counts[item] = 0
                        total_scores[item] += rating
                        rating_counts[item] += 1
                current = current.next

        average_scores = {}
        for item in total_scores:
            average_scores[item] = round(total_scores[item] / rating_counts[item], 2)

        print("Average scores for all items:")
        for item, score in average_scores.items():
            formatted_item = item[:4] + " " + item[4:]  
            print(f"{formatted_item}: {score}")

        best_item = None
        highest_avg_score = 0
        for item, avg_score in average_scores.items():
            if avg_score > highest_avg_score:
                highest_avg_score = avg_score
                best_item = item

        if best_item is None:
            return "No recommendations available."
        formatted_best_item = best_item[:4] + " " + best_item[4:]
        return (formatted_best_item, highest_avg_score)

rec_sys = Recommender()

rec_sys.add_ratings("User1", {"ItemA": 5, "ItemB": 3, "ItemC": 4, "ItemD": 2, "ItemE": 1})
rec_sys.add_ratings("User2", {"ItemA": 3, "ItemB": 5, "ItemC": 1, "ItemD": 4, "ItemE": 2})
rec_sys.add_ratings("User3", {"ItemA": 4, "ItemB": 2, "ItemC": 5, "ItemD": 1, "ItemE": 3})
rec_sys.add_ratings("User4", {"ItemA": 2, "ItemB": 4, "ItemC": 3, "ItemD": 5, "ItemE": 1})
rec_sys.add_ratings("User5", {"ItemA": 1, "ItemB": 3, "ItemC": 4, "ItemD": 2, "ItemE": 5})
rec_sys.add_ratings("User6", {"ItemA": 5, "ItemB": 1, "ItemC": 3, "ItemD": 4, "ItemE": 2})
rec_sys.add_ratings("User7", {"ItemA": 3, "ItemB": 4, "ItemC": 2, "ItemD": 5, "ItemE": 1})

recommendation = rec_sys.recommend("User1")
print("\nRecommendation for User1:")
print(f"Item: {recommendation[0]}, Score: {recommendation[1]:.2f}")
