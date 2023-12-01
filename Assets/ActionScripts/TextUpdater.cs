using UnityEngine;
using TMPro;
using UnityEngine.Networking;
using System.Collections;

[System.Serializable]
public class TargetTextResponse
{
    public string target_text;
    public string confidence;
}

public class TextUpdater : MonoBehaviour
{
    public TextMeshPro targetText;

    private readonly string backendUrl = "http://34.130.78.182/get_target_text";

    private void Start()
    {
        InvokeRepeating("UpdateText", 0f, 1f);
    }

    void UpdateText()
    {
        StartCoroutine(FetchAndUpdateText());
    }

    IEnumerator FetchAndUpdateText()
    {
        UnityWebRequest request = UnityWebRequest.Get(backendUrl);
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonResponse = request.downloadHandler.text;
            TargetTextResponse response = JsonUtility.FromJson<TargetTextResponse>(jsonResponse);

            if (response != null)
            {
                string newText = response.target_text;
                targetText.text = newText;
                string confidence = response.confidence;
                var textColor = confidence switch
                {
                    "high" => Color.green,
                    "mid" => new Color(1.0f, 0.5f, 0.0f),// Orange
                    "low" => Color.red,
                    _ => Color.black,// default black
                };
                targetText.color = textColor;
            }
            else
            {
                Debug.LogError("Failed to parse JSON response.");
            }
        }
        else
        {
            Debug.LogError("Failed to fetch data from the backend.");
        }
    }
}
