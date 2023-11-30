import random
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


def train_sentence_transformer(model, train_examples, validation_examples, batch_size, num_epochs):
    # Define the DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # Define the loss function
    train_loss = losses.CosineSimilarityLoss(model)

    # Prepare test data for evaluation
    test_sentences1 = [example.texts[0] for example in validation_examples]
    test_sentences2 = [example.texts[1] for example in validation_examples]
    test_scores = [example.label for example in validation_examples]

    # Define the evaluation metric
    evaluator = EmbeddingSimilarityEvaluator(test_sentences1, test_sentences2, test_scores)

    # Fine-tune the model
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path="output",
    )
    return model


def evaluate(model, test_examples, batch_size=8):
    test_sentences1 = [example.texts[0] for example in test_examples]
    test_sentences2 = [example.texts[1] for example in test_examples]
    test_scores = [example.label for example in test_examples]

    # Compute embeddings for test sentences
    embeddings1 = model.encode(test_sentences1, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
    embeddings2 = model.encode(test_sentences2, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)

    # Move embeddings to CPU and convert to NumPy arrays
    embeddings1 = embeddings1.cpu().numpy()
    embeddings2 = embeddings2.cpu().numpy()

    # Compute similarity scores
    cosine_scores = np.inner(embeddings1, embeddings2)

    # Flatten the cosine_scores array
    cosine_scores_flat = np.diag(cosine_scores)

    # Compute Pearson correlation
    pearson_correlation, _ = pearsonr(test_scores, cosine_scores_flat)

    # Compute Spearman correlation
    spearman_correlation, _ = spearmanr(test_scores, cosine_scores_flat)

    # Compute Mean Squared Error
    mse = mean_squared_error(test_scores, cosine_scores_flat)

    return pearson_correlation, spearman_correlation, mse


if __name__ == '__main__':
    input_data = [
        {
            "sentence1": "The largest Tesco store in the United Kingdom is located in Kingston Park on the edge of Newcastle.",
            "sentence2": "Close to Newcastle, the largest indoor shopping centre in Europe, the MetroCentre, is located in Gateshead.",
            "similarity score": 2},
        {"sentence1": "The largest girls' independent school is Newcastle High School for Girls.",
         "sentence2": "Newcastle School for Boys is the only independent boys' only school in the city and is situated in Gosforth.",
         "similarity score": 2},
        {
            "sentence1": "The immune system protects organisms from infection with layered defenses of increasing specificity.",
            "sentence2": "In simple terms, physical barriers prevent pathogens such as bacteria and viruses from entering the organism.",
            "similarity score": 4},
        {
            "sentence1": "The Scots king William the Lion was imprisoned in Newcastle in 1174, and Edward I brought the Stone of Scone and William Wallace south through the town.",
            "sentence2": "The Grainger Market replaced an earlier market originally built in 1808 called the Butcher Market. ",
            "similarity score": 2},
        {
            "sentence1": "The opening was celebrated with a grand dinner attended by 2000 guests, and the Laing Art Gallery has a painting of this event.",
            "sentence2": "Using boiling water to produce mechanical motion goes back over 2000 years, but early devices were not practical.",
            "similarity score": 0},
        {
            "sentence1": "The immune system is a system of many biological structures and processes within an organism that protects against disease.",
            "sentence2": "In many species, the immune system can be classified into subsystems, such as the innate immune system versus the adaptive immune system, or humoral immunity versus cell-mediated immunity. ",
            "similarity score": 1},
        {"sentence1": "Some words used in the Geordie dialect are used elsewhere in the Northern United Kingdom.",
         "sentence2": "The words 'bonny' (meaning 'pretty'), 'howay' ('come on'), 'stot' ('bounce') and 'hadaway' ('go away' or 'you're kidding'), all appear to be used in Scots; 'aye' ('yes') and 'nowt' (IPA://naʊt/, rhymes with out,'nothing') are used elsewhere in Northern England.",
         "similarity score": 3},
        {"sentence1": "The 2010 United States Census reported that Fresno had a population of 494,665.",
         "sentence2": "The population density was 4,404.5 people per square mile (1,700.6/km²).",
         "similarity score": 3},
        {
            "sentence1": "Fresno is marked by a semi-arid climate (Köppen BSh), with mild, moist winters and hot and dry summers, thus displaying Mediterranean characteristics.",
            "sentence2": "Summers provide considerable sunshine, with July peaking at 97 percent of the total possible sunlight hours; conversely, January is the lowest with only 46 percent of the daylight time in sunlight because of thick tule fog. However, the year averages 81% of possible sunshine, for a total of 3550 hours. Average annual precipitation is around 11.5 inches (292.1 mm), which, by definition, would classify the area as a semidesert.",
            "similarity score": 4},
        {
            "sentence1": "Fresno meteorology was selected in a national U.S. Environmental Protection Agency study for analysis of equilibrium temperature for use of ten-year meteorological data to represent a warm, dry western United States locale.",
            "sentence2": "The area is also known for its early twentieth century homes, many of which have been restored in recent decades.",
            "similarity score": 0},
        {"sentence1": "The historic heart of Newcastle is the Grainger Town area.",
         "sentence2": "Established on classical streets built by Richard Grainger, a builder and developer, between 1835 and 1842, some of Newcastle upon Tyne's finest buildings and streets lie within this area of the city centre including Grainger Market, Theatre Royal, Grey Street, Grainger Street and Clayton Street.",
         "similarity score": 4},
        {
            "sentence1": "Between the 1880s and World War II, Downtown Fresno flourished, filled with electric Street Cars, and contained some of the San Joaquin Valley's most beautiful architectural buildings.",
            "sentence2": "Among them, the original Fresno County Courthouse (demolished), the Fresno Carnegie Public Library (demolished), the Fresno Water Tower, the Bank of Italy Building, the Pacific Southwest Building, the San Joaquin Light & Power Building (currently known as the Grand 1401), and the Hughes Hotel (burned down), to name a few.",
            "similarity score": 4},
        {
            "sentence1": "Disorders of the immune system can result in autoimmune diseases, inflammatory diseases and cancer.",
            "sentence2": "Immunodeficiency occurs when the immune system is less active than normal, resulting in recurring and life-threatening infections.",
            "similarity score": 4},
        {
            "sentence1": "Using boiling water to produce mechanical motion goes back over 2000 years, but early devices were not practical.",
            "sentence2": "Other components are often present; pumps (such as an injector) to supply water to the boiler during operation, condensers to recirculate the water and recover the latent heat of vaporisation, and superheaters to raise the temperature of the steam above its saturated vapour point, and various mechanisms to increase the draft for fireboxes.",
            "similarity score": 1},
        {"sentence1": "The result is the multiple expansion engine.",
         "sentence2": "Such engines use either three or four expansion stages and are known as triple and quadruple expansion engines respectively.",
         "similarity score": 4},
        {
            "sentence1": "This vibrant and culturally diverse area of retail businesses and residences experienced a renewal after a significant decline in the late 1960s and 1970s.",
            "sentence2": "After decades of neglect and suburban flight, the neighborhood revival followed the re-opening of the Tower Theatre in the late 1970s, which at that time showed second and third run movies, along with classic films.",
            "similarity score": 4},
        {
            "sentence1": "The Northern Rock Cyclone, a cycling festival, takes place within, or starting from, Newcastle in June. ",
            "sentence2": "The Northern Pride Festival and Parade is held in Leazes Park and in the city's Gay Community in mid July.",
            "similarity score": 2},
        {
            "sentence1": "There are concentrations of pubs, bars and nightclubs around the Bigg Market and the Quayside area of the city centre. ",
            "sentence2": "A B cell identifies pathogens when antibodies on its surface bind to a specific foreign antigen.",
            "similarity score": 0},
        {
            "sentence1": "There are 3 main bus companies providing services in the city; Arriva North East, Go North East and Stagecoach North East.",
            "sentence2": "There are two major bus stations in the city: Haymarket bus station and Eldon Square bus station.",
            "similarity score": 2},
        {
            "sentence1": "There are 3 main bus companies providing services in the city; Arriva North East, Go North East and Stagecoach North East.",
            "sentence2": "Stagecoach is the primary operator in the city proper, with cross-city services mainly between both the West and East ends via the city centre with some services extending out to the MetroCentre, Killingworth, Wallsend and Ponteland. ",
            "similarity score": 3},
    ]

    # Split the dataset into train, validation and test
    random.shuffle(input_data)
    train_split_index = int(len(input_data) * 0.7)
    test_split_index = int(len(input_data) * 0.85)
    train_data = input_data[:train_split_index]
    test_data = input_data[train_split_index:test_split_index]
    validation_data = input_data[test_split_index:]

    train_examples = [
        InputExample(texts=[data["sentence1"], data["sentence2"]], label=data["similarity score"] / 5)
        for data in train_data
    ]

    validation_examples = [
        InputExample(texts=[data["sentence1"], data["sentence2"]], label=data["similarity score"] / 5)
        for data in validation_data
    ]

    test_examples = [
        InputExample(texts=[data["sentence1"], data["sentence2"]], label=data["similarity score"] / 5)
        for data in test_data
    ]

    model = SentenceTransformer("sentence-transformers/paraphrase-distilroberta-base-v1")
    batch_size = 4
    num_epochs = 4
    fine_tuned_model = train_sentence_transformer(model, train_examples, validation_examples, batch_size, num_epochs)
    fine_tuned_model.save("Web/similarity_model")

    # evaluation
    pearson_correlation_orig, spearman_correlation_orig, mse_orig = evaluate(model, test_examples, batch_size)
    pearson_correlation, spearman_correlation, mse = evaluate(fine_tuned_model, test_examples, batch_size)

    print(f"Original Pearson Correlation: {pearson_correlation_orig:.4f}")
    print(f"Original Spearman Correlation: {spearman_correlation_orig:.4f}")
    print(f"Original mse: {mse_orig:.4f}")

    print(f"Pearson Correlation: {pearson_correlation:.4f}")
    print(f"Spearman Correlation: {spearman_correlation:.4f}")
    print(f"mse: {mse:.4f}")
