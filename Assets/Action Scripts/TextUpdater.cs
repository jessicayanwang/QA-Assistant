using UnityEngine;
using TMPro;
using UnityEngine.Networking;
using System.Collections;

[System.Serializable]
public class TargetTextResponse
{
    public string target_text;
}

public class TextUpdater : MonoBehaviour
{
    public TextMeshProUGUI targetText;

    private readonly string backendUrl = "http://127.0.0.1:5000/get_target_text";

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
