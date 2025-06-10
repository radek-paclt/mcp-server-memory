import { z } from 'zod';

export function zodToJsonSchema(schema: z.ZodType<any>): any {
  // Simple conversion for basic types
  if (schema instanceof z.ZodString) {
    return { type: 'string' };
  }
  if (schema instanceof z.ZodNumber) {
    return { type: 'number' };
  }
  if (schema instanceof z.ZodBoolean) {
    return { type: 'boolean' };
  }
  if (schema instanceof z.ZodArray) {
    return {
      type: 'array',
      items: zodToJsonSchema(schema._def.type),
    };
  }
  if (schema instanceof z.ZodObject) {
    const shape = schema._def.shape();
    const properties: any = {};
    const required: string[] = [];

    for (const [key, value] of Object.entries(shape)) {
      if (value instanceof z.ZodOptional) {
        properties[key] = zodToJsonSchema(value._def.innerType);
      } else {
        properties[key] = zodToJsonSchema(value as z.ZodType<any>);
        required.push(key);
      }
    }

    return {
      type: 'object',
      properties,
      required: required.length > 0 ? required : undefined,
    };
  }
  if (schema instanceof z.ZodEnum) {
    return {
      type: 'string',
      enum: schema._def.values,
    };
  }
  if (schema instanceof z.ZodOptional) {
    return zodToJsonSchema(schema._def.innerType);
  }
  if (schema instanceof z.ZodDefault) {
    const jsonSchema = zodToJsonSchema(schema._def.innerType);
    jsonSchema.default = schema._def.defaultValue();
    return jsonSchema;
  }

  // Fallback
  return { type: 'string' };
}