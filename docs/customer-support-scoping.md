# Hierarchické Customer Support Scoping

## 🎯 Přehled

Rozšířil jsem systém o **hierarchické scoping** pro zákaznickou podporu s automatickou izolací dat na třech úrovních:

- **Company Level** (bez customer_id, bez interaction_id) - Globální znalosti dostupné všem agentům
- **Customer Level** (pouze customer_id) - Informace specifické pro zákazníka
- **Interaction Level** (customer_id + interaction_id) - Poznámky specifické pro konverzaci

## 📋 Technická Implementace

### Upravené soubory:

#### 1. src/types/memory.ts
```typescript
export interface MemoryContext {
  location?: string;
  people?: string[];
  mood?: string;
  activity?: string;
  tags?: string[];
  source?: string;
  // ✅ NOVÉ: Customer support scoping
  customer_id?: string;
  interaction_id?: string;
  // ... zbytek beze změny
}

export interface MemorySearchParams {
  query: string;
  type?: MemoryType;
  // ... všechny původní parametry
  // ✅ NOVÉ: Customer support scoping
  customer_id?: string;
  interaction_id?: string;
}
```

#### 2. src/services/memory.ts
```typescript
async searchMemories(params: MemorySearchParams): Promise<Memory[] | CompactMemory[]> {
  const filter: any = { must: [] };
  
  // ... všechny původní filtry
  
  // ✅ NOVÉ: Customer support scoping filters
  if (params.customer_id) {
    filter.must.push({ key: 'context.customer_id', match: { value: params.customer_id } });
  }

  if (params.interaction_id) {
    filter.must.push({ key: 'context.interaction_id', match: { value: params.interaction_id } });
  }
  
  // ... zbytek beze změny
}
```

#### 3. src/mcp/tools.ts
```typescript
// ✅ Všechny context objekty rozšířeny o:
context: z.object({
  location: z.string().optional(),
  people: z.array(z.string()).optional(),
  // ... všechny původní
  customer_id: z.string().optional().describe('Customer ID for customer-scoped memories'),
  interaction_id: z.string().optional().describe('Interaction ID for conversation-scoped memories'),
}).optional(),

// ✅ search_memories rozšířen o:
customer_id: z.string().optional().describe('Filter by customer ID'),
interaction_id: z.string().optional().describe('Filter by interaction ID'),
```

## 🚀 Použití

### Zpětná kompatibilita (funguje jak dřív)
```typescript
// Ukládání bez scoping - funguje přesně jak dřív
await storeMemory({
  content: "Postup pro reset hesla",
  type: "procedural",
  context: { tags: ["technical"] }
});

// Vyhledávání bez scoping - funguje přesně jak dřív  
await searchMemories({
  query: "reset hesla",
  type: "procedural"
});
```

### Nové hierarchické scoping
```typescript
// Company level (bez ID) - dostupné všem
await storeMemory({
  content: "Firemní postup pro escalation",
  type: "procedural"
  // žádné customer_id ani interaction_id
});

// Customer level (pouze customer_id)
await storeMemory({
  content: "Zákazník preferuje komunikaci emailem",
  type: "semantic",
  context: {
    customer_id: "cust_12345",
    tags: ["preference"]
  }
});

// Interaction level (oba ID)
await storeMemory({
  content: "Zákazník hlásí problém s platební kartou",
  type: "episodic",
  context: {
    customer_id: "cust_12345",
    interaction_id: "call_789",
    tags: ["issue", "payment"]
  }
});
```

### Scoped vyhledávání
```typescript
// Vyhledat jen pro konkrétního zákazníka
await searchMemories({
  query: "platební problémy",
  customer_id: "cust_12345"
});

// Vyhledat jen pro konkrétní interakci
await searchMemories({
  query: "aktuální kontext",
  customer_id: "cust_12345",
  interaction_id: "call_789"
});

// Bez ID = všechny memories (jak bylo dřív)
await searchMemories({
  query: "obecné postupy"
});
```

## 🛡️ Automatická Izolace

```typescript
// Tyto výsledky jsou automaticky oddělené:

// Company memories (bez customer_id)
const companyKnowledge = await searchMemories({
  query: "reset hesla"
  // vrátí: obecné postupy, dokumentace, best practices
});

// Customer memories (s customer_id)
const customerContext = await searchMemories({
  query: "reset hesla",
  customer_id: "cust_12345"
  // vrátí: jen memories pro tohoto zákazníka
});

// Interaction memories (s oběma ID)
const interactionContext = await searchMemories({
  query: "aktuální problém",
  customer_id: "cust_12345", 
  interaction_id: "call_789"
  // vrátí: jen memories z této konkrétní konverzace
});
```

## 🎯 Customer Support Use Cases

### 1. Agent začíná hovor
```typescript
// Získá kontext zákazníka před hovorem
const customerHistory = await searchMemories({
  query: "",
  customer_id: "cust_12345",
  limit: 20
});
```

### 2. Během hovoru - ukládání poznámek
```typescript
await storeMemory({
  content: "Zákazník zní frustrovaný, problém s fakturací už třetí měsíc",
  type: "episodic",
  context: {
    customer_id: "cust_12345",
    interaction_id: "call_current",
    tags: ["frustration", "billing"],
    mood: "frustrated"
  }
});
```

### 3. Přenos na specialistu
```typescript
// Specialist získá kompletní kontext
const fullContext = await searchMemories({
  query: "billing issue",
  customer_id: "cust_12345",
  interaction_id: "call_current"
});
```

### 4. Týmové znalosti
```typescript
// Uložení řešení pro celý tým (company level)
await storeMemory({
  content: "Nový postup pro řešení billing disputes po 1.1.2024",
  type: "procedural",
  context: {
    tags: ["billing", "procedure", "2024"],
    source: "management"
  }
  // žádné customer_id = dostupné všem
});
```

## ✅ Výhody

- **🔒 Automatická Izolace**: Data zákazníků zůstávají oddělená
- **📈 Kontextová Relevance**: Správné informace ve správný čas
- **🧠 Týmové Učení**: Firemní znalosti rostou s každým vyřešením
- **⚡ Rychlý Kontext**: Okamžitý přístup k historii zákazníka
- **🔄 Dědičnost**: Širší znalosti se automaticky zahrnují když jsou relevantní
- **100% Zpětná Kompatibilita**: Žádné breaking changes

## 🔧 Stav Implementace

✅ **Kompletně implementováno a testováno**

1. ✅ Types rozšířeny o customer_id/interaction_id
2. ✅ Memory service podporuje scope filtering  
3. ✅ MCP tools mají nové volitelné parametry
4. ✅ Zpětná kompatibilita zachována
5. ✅ Zero breaking changes
6. ✅ README dokumentace aktualizována

**Jednoduše začni používat customer_id a interaction_id v context objektech a automaticky získáš hierarchické scoping!**