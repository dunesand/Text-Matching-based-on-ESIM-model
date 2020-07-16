
import esim_model

model = ESIM(hidden_size=256,embeds_dim=embedding_dim,linear_size=256).cuda()
model.fit(save_path="./model_file/esim_model.pkl")
