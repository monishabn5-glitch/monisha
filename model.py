import torch
import torch.nn as nn
from torchvision import models

CFG = {
    "feat_dim": 256,
    "lstm_hidden": 128
}

class SpatialEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        # ✅ FIX 1: Load pretrained ResNet18 weights from ImageNet
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.proj(self.features(x))

class CrossRegionConsistency(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, f, e, m):
        diff_fe = torch.abs(f - e)
        diff_em = torch.abs(e - m)
        diff_fm = torch.abs(f - m)
        return self.fc(torch.cat([diff_fe, diff_em, diff_fm], dim=-1))

class AttentionFusion(nn.Module):
    def __init__(self, feat_dim, num_features=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_features = num_features
        
        # ✅ FIX 2: Improved attention network
        self.attn_net = nn.Sequential(
            nn.Linear(feat_dim * num_features, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_features),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, f, e, m, c):
        # Concatenate all features
        concat = torch.cat([f, e, m, c], dim=-1)
        
        # Compute attention weights
        weights = self.attn_net(concat)
        
        # Stack features
        stacked = torch.stack([f, e, m, c], dim=1)
        
        # Apply weights properly
        weighted = stacked * weights.unsqueeze(-1)
        fused = weighted.sum(dim=1)
        
        return fused

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_face = SpatialEncoder(CFG["feat_dim"])
        self.enc_eye = SpatialEncoder(CFG["feat_dim"])
        self.enc_mouth = SpatialEncoder(CFG["feat_dim"])
        
        self.lstm_face = nn.LSTM(CFG["feat_dim"], CFG["lstm_hidden"], batch_first=True, dropout=0.2)
        self.lstm_eye = nn.LSTM(CFG["feat_dim"], CFG["lstm_hidden"], batch_first=True, dropout=0.2)
        self.lstm_mouth = nn.LSTM(CFG["feat_dim"], CFG["lstm_hidden"], batch_first=True, dropout=0.2)
        
        self.consistency = CrossRegionConsistency(CFG["lstm_hidden"])
        self.fusion = AttentionFusion(CFG["lstm_hidden"], num_features=4)
        
        # ✅ FIX 3: Improved classifier with better regularization
        self.classifier = nn.Sequential(
            nn.Linear(CFG["lstm_hidden"], 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def process_stream(self, seq, encoder, lstm):
        """
        Process a sequence through encoder and LSTM.
        ✅ FIX 4: Use LSTM hidden state instead of last output
        """
        B, T, C, H, W = seq.shape
        
        # Extract spatial features for all frames
        feats = encoder(seq.view(B * T, C, H, W)).view(B, T, -1)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = lstm(feats)
        
        # Use hidden state (cleaner than last output)
        lstm_final = h_n.squeeze(0)
        
        return lstm_final

    def forward(self, face, eye, mouth):
        """
        Forward pass through the model.
        
        Args:
            face: (B, T, C, 224, 224)
            eye: (B, T, C, 112, 112)
            mouth: (B, T, C, 112, 112)
        """
        f_feat = self.process_stream(face, self.enc_face, self.lstm_face)
        e_feat = self.process_stream(eye, self.enc_eye, self.lstm_eye)
        m_feat = self.process_stream(mouth, self.enc_mouth, self.lstm_mouth)
        
        # Cross-region consistency
        c_feat = self.consistency(f_feat, e_feat, m_feat)
        
        # Attention fusion
        fused_feat = self.fusion(f_feat, e_feat, m_feat, c_feat)
        
        # Final classification
        logits = self.classifier(fused_feat)
        
        return logits
