import json
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MOTEUR, SORTIE = sys.argv[1], sys.argv[2]

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)
moteur = AutoModelForCausalLM.from_pretrained(MOTEUR, torch_dtype=torch.bfloat16, device_map="cuda")
moteur.gradient_checkpointing_enable() # stope les paramètres des matrices internes de l'attention de ne pas être calculés pour économiser la mémoire
moteur.enable_input_require_grads() # stope les paramètres de la matrice d'embedding (d'entrées) de ne pas être calculés pour économiser la mémoire

# la fonction get_peft_model() va créer un adaptateur LoRA pour le moteur et ne va pas modifier les poids du moteur.
# L'adaptateur va être entraîné et les poids du moteur vont rester inchangés.
# on va entraîner que les matrices de projection q_proj et v_proj de l'attention, qui sont les plus importantes pour la qualité du modèle.
moteur = get_peft_model(moteur, LoraConfig(r=16, target_modules=["q_proj", "v_proj", "o_proj", "k_proj"], task_type="CAUSAL_LM", lora_alpha=32, lora_dropout=0.05, use_rslora=True))
entraines = [p for p in moteur.parameters() if p.requires_grad]
for p in entraines:
    p.data = p.data.float() # p (un tenseur). Pour accèder aux valeurs numériques d'un tenseurs on utilise p.data
print("paramètres entraînés", sum(p.numel() for p in entraines))


def exemple(paire):
    demande = tokeniseur.apply_chat_template(
        [{"role": "user", "content": paire["demande"]}], tokenize=False, add_generation_prompt=True)
    ids_demande = tokeniseur(demande)["input_ids"]
    ids = tokeniseur(demande + paire["reponse"] + "<|im_end|>")["input_ids"]
    cibles = [-100] * len(ids_demande) + ids[len(ids_demande):]
    return torch.tensor([ids]).cuda(), torch.tensor([cibles]).cuda()



# Commencer l'entraînement de l'adaptateur LoRA sur les paires de demande-réponse.
paires = json.load(open("paires.json", encoding="utf-8"))
adam = torch.optim.AdamW(entraines, lr=2e-4)
random.seed(0)
for epoque in range(3):
    random.shuffle(paires)
    pertes = []
    # on part element par element (stockastique) pour que la perte soit plus stable et que l'adaptateur converge mieux.
    for paire in paires:
        ids, cibles = exemple(paire)
        # labels donne la position du vrai token à prédire pour calculer la perte. La perte est calculée sur les tokens de la réponse et pas sur les tokens de la demande.
        # On met -100 pour les tokens de la demande pour que la perte ne soit pas calculée sur ces tokens. (il n y a aucun token avec la valeur -100)
        # On ajoute le parametre labels pour acceder à la cible (à lauqelle on a appliqué le masque) et on pas à "ids" (la demande réelle)
        perte = moteur(input_ids=ids, labels=cibles).loss 
        perte.backward()

        torch.nn.utils.clip_grad_norm_(
            entraines,
            max_norm=1.0
        )  # pour éviter que les gradients explosent et que l'adaptateur diverge. On limite la norme des gradients à 1.0


        adam.step()
        adam.zero_grad() # ne pas stocker les gradients (c'est facultatif, ils ne prennent pas beacoup de mémoire)
        pertes.append(perte.item())
    print(f"époque {epoque + 1}, perte moyenne {sum(pertes) / len(pertes):.3f}")

for p in entraines:
    p.data = p.data.to(torch.float16)
moteur.save_pretrained(SORTIE) 
print("adaptateur enregistré dans le dossier", SORTIE)
