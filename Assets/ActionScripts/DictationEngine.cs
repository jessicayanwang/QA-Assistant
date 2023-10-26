using UnityEngine;
using UnityEngine.UI;
using Microsoft.CognitiveServices.Speech;
using System.Collections;
using PimDeWitte.UnityMainThreadDispatcher;
using UnityEngine.Networking;
using TMPro;



public class DictationEngine : MonoBehaviour
{
    public TextMeshPro outputText;
    public Button startRecoButton;

    private readonly object threadLocker = new();
    private bool waitingForReco;
    private string message;

    private bool micPermissionGranted = false;

    private UnityMainThreadDispatcher dispatcher; // Reference to the UnityMainThreadDispatcher

    public async void ButtonClick()
    {
        // Creates an instance of a speech config with specified subscription key and service region.
        // Replace with your own subscription key and service region (e.g., "westus").
        var config = SpeechConfig.FromSubscription("8a122857214847c9bc1c530227a3eac8", "canadacentral");

        // Make sure to dispose the recognizer after use!
        using var recognizer = new SpeechRecognizer(config);
        lock (threadLocker)
        {
            waitingForReco = true;
        }

        // Starts speech recognition, and returns after a single utterance is recognized.
        var result = await recognizer.RecognizeOnceAsync().ConfigureAwait(false);

        // Checks result.
        string newMessage = string.Empty;
        if (result.Reason == ResultReason.RecognizedSpeech)
        {
            newMessage = result.Text;
            // Send the recognized text to Flask backend 
            dispatcher.Enqueue(SendToBackend(newMessage));
        }
        else if (result.Reason == ResultReason.NoMatch)
        {
            newMessage = "NOMATCH: Speech could not be recognized.";
        }
        else if (result.Reason == ResultReason.Canceled)
        {
            var cancellation = CancellationDetails.FromResult(result);
            newMessage = $"CANCELED: Reason={cancellation.Reason} ErrorDetails={cancellation.ErrorDetails}";
        }

        lock (threadLocker)
        {
            message = newMessage;
            waitingForReco = false;
        }
    }

    public IEnumerator SendToBackend(string text)
    {
        string backendUrl = "http://127.0.0.1:5000/set_target_text";

        TargetTextResponse jsonData = new()
        {
            target_text = text
        };
        string jsonDataString = JsonUtility.ToJson(jsonData);

        var uwr = new UnityWebRequest(backendUrl, "POST");
        byte[] bodyRaw = new System.Text.UTF8Encoding().GetBytes(jsonDataString);
        uwr.uploadHandler = (UploadHandler)new UploadHandlerRaw(bodyRaw);
        uwr.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
        uwr.SetRequestHeader("Content-Type", "application/json");

        yield return uwr.SendWebRequest();

        if (uwr.result == UnityWebRequest.Result.ConnectionError)
        {
            dispatcher.Enqueue(() => Debug.LogError("Error sending data to backend: " + uwr.error));
        }
        else if (uwr.result == UnityWebRequest.Result.Success)
        {
            dispatcher.Enqueue(() => Debug.Log("Data sent to backend successfully"));
        }

        // Use the dispatcher to update the UI from the main thread
        dispatcher.Enqueue(() => {
            if (outputText != null)
            {
                outputText.text = message;
            }
        });
    }

    void Start()
    {
        dispatcher = UnityMainThreadDispatcher.Instance(); // Get the instance of the dispatcher

        if (outputText == null)
        {
            Debug.LogError("outputText property is null! Assign a UI Text element to it.");
        }
        else if (startRecoButton == null)
        {
            message = "startRecoButton property is null! Assign a UI Button to it.";
            Debug.LogError(message);
        }
        else
        {
            // Continue with normal initialization, Text and Button objects are present.

            micPermissionGranted = true;
            message = "Click button to recognize speech";

            startRecoButton.onClick.AddListener(ButtonClick);
        }
    }

    void Update()
    {
#if PLATFORM_ANDROID
        if (!micPermissionGranted && Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            micPermissionGranted = true;
            message = "Click button to recognize speech";
        }
#elif PLATFORM_IOS
        if (!micPermissionGranted && Application.HasUserAuthorization(UserAuthorization.Microphone))
        {
            micPermissionGranted = true;
            message = "Click button to recognize speech";
        }
#endif

        lock (threadLocker)
        {
            if (startRecoButton != null)
            {
                startRecoButton.interactable = !waitingForReco && micPermissionGranted;
            }
            if (outputText != null)
            {
                outputText.text = message;
            }
        }
    }
}
// </code>