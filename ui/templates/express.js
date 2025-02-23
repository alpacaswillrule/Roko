import express from 'express';
import cors from 'cors';  
import {
    Client,
    PrivateKey,
    AccountId,
    TransferTransaction,
    Hbar,
    HbarUnit,
    AccountBalanceQuery,
} from '@hashgraph/sdk';
import dotenv from 'dotenv';
// import fetch from 'node-fetch';

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

const port = 3000;
let client;

app.post('/transferHbar', async (req, res) => {
    try {
        console.log(req.body)
        const {cost} = req.body;

        const operatorIdStr = process.env.OPERATOR_ACCOUNT_ID;
        const operatorKeyStr = process.env.OPERATOR_ACCOUNT_PRIVATE_KEY;
        const minerIdStr = process.env.ACCOUNT_0_ID

        if (!operatorIdStr || !operatorKeyStr || ! minerIdStr) {
            throw new Error('Missing OPERATOR_ACCOUNT_ID or OPERATOR_ACCOUNT_PRIVATE_KEY');
        }

        const operatorId = AccountId.fromString(operatorIdStr);
        const minerId = AccountId.fromString(minerIdStr);
        const operatorKey = PrivateKey.fromStringECDSA(operatorKeyStr);

        client = Client.forTestnet().setOperator(operatorId, operatorKey);
        client.setDefaultMaxTransactionFee(new Hbar(100));
        client.setDefaultMaxQueryPayment(new Hbar(50));

        const currentDateTime = new Date().toLocaleString(); // Get current date and time
        const memoMessage = `Query transaction at Roko at ${currentDateTime}`;
        
        // 🔹 Perform Transfer
        const transferTx = await new TransferTransaction()
            .setTransactionMemo(memoMessage)
            .addHbarTransfer(operatorId, new Hbar(-1*cost, HbarUnit.Hbar))
            .addHbarTransfer(minerId, new Hbar(0.4 * cost, HbarUnit.Hbar))
            .addHbarTransfer('0.0.5615771', new Hbar(0.6 * cost, HbarUnit.Hbar))
            .freezeWith(client);

        const transferTxSigned = await transferTx.sign(operatorKey);
        const transferTxSubmitted = await transferTxSigned.execute(client);
        const transferTxReceipt = await transferTxSubmitted.getReceipt(client);
        const transactionStatus = transferTxReceipt.status;

        // 🔹 Get Account Balance
        const newAccountBalance = await new AccountBalanceQuery()
            .setAccountId(operatorId)
            .execute(client);

        const newHbarBalance = newAccountBalance.hbars;

        // 🔹 Transaction Verification URL
        const transferTxId = transferTx.transactionId;
        const transferTxVerifyUrl = `https://hashscan.io/testnet/transaction/${transferTxId}`;

        client.close();

        res.json({
            status: transactionStatus.toString(),
            // balance: newHbarBalance.toString(),
            transactionUrl: transferTxVerifyUrl,
        });

    } catch (ex) {
        client && client.close();
        res.status(500).json({ error: ex.message });
    }
});

app.get('/getUserBalance', async (req, res) => {
    try {
        const operatorIdStr = process.env.OPERATOR_ACCOUNT_ID;
        const operatorKeyStr = process.env.OPERATOR_ACCOUNT_PRIVATE_KEY;

        if (!operatorIdStr || !operatorKeyStr) {
            throw new Error('Missing ID or PK');
        }

        const operatorId = AccountId.fromString(operatorIdStr);
        const operatorKey = PrivateKey.fromStringECDSA(operatorKeyStr);

        client = Client.forTestnet().setOperator(operatorId, operatorKey);
        client.setDefaultMaxTransactionFee(new Hbar(100));
        client.setDefaultMaxQueryPayment(new Hbar(50));

        // 🔹 Get Account Balance
        const accountBalance = await new AccountBalanceQuery()
            .setAccountId(operatorId)
            .execute(client);

        const hbarBalance = accountBalance.hbars.toString();

        client.close();

        res.json({
            status: 'success',
            balance: hbarBalance,
        });

    } catch (ex) {
        client && client.close();
        res.status(500).json({ error: ex.message });
    }
});


app.get('/getMinerBalance', async (req, res) => {
    try {
        const operatorIdStr = process.env.ACCOUNT_0_ID;
        const operatorKeyStr = process.env.ACCOUNT_0_PRIVATE_KEY;

        if (!operatorIdStr || !operatorKeyStr) {
            throw new Error('Missing ID or PK');
        }

        const operatorId = AccountId.fromString(operatorIdStr);
        const operatorKey = PrivateKey.fromStringECDSA(operatorKeyStr);

        client = Client.forTestnet().setOperator(operatorId, operatorKey);
        client.setDefaultMaxTransactionFee(new Hbar(100));
        client.setDefaultMaxQueryPayment(new Hbar(50));

        // 🔹 Get Account Balance
        const accountBalance = await new AccountBalanceQuery()
            .setAccountId(operatorId)
            .execute(client);

        const hbarBalance = accountBalance.hbars.toString();

        client.close();

        res.json({
            status: 'success',
            balance: hbarBalance,
        });

    } catch (ex) {
        client && client.close();
        res.status(500).json({ error: ex.message });
    }
});


// Simple health check endpoint
app.get("/", (req, res) => {
    res.send("Plz work");
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
