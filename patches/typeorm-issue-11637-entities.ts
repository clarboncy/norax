/**
 * Entity definitions for TypeORM issue #11637 regression test
 * Place at: test/github-issues/11637/entity/entities.ts
 */

import { EntitySchema } from "../../../../../src/entity-schema/EntitySchema"

// Specialty entity
export const specialtyEntitySchema = new EntitySchema({
    name: "Specialty",
    tableName: "specialty",
    columns: {
        name: {
            type: "varchar",
            primary: true,
        },
    },
})

// VetSpecialty — user-defined junction entity
export const vetSpecialtyEntitySchema = new EntitySchema({
    name: "VetSpecialty",
    tableName: "vet_specialty",
    columns: {
        id: {
            type: "int",
            primary: true,
            generated: "increment",
        },
        vetId: {
            type: "int",
            name: "vet_id",
        },
        specialtyName: {
            type: "varchar",
            name: "specialty_name",
        },
    },
    relations: {
        vet: {
            type: "many-to-one",
            target: "Vet",
            joinColumn: {
                name: "vet_id",
                referencedColumnName: "id",
            },
        },
        specialty: {
            type: "many-to-one",
            target: "Specialty",
            joinColumn: {
                name: "specialty_name",
                referencedColumnName: "name",
            },
        },
    },
})

// Vet entity with many-to-many to Specialty via user-defined junction
export const vetEntitySchema = new EntitySchema({
    name: "Vet",
    tableName: "vet",
    columns: {
        id: {
            type: "int",
            primary: true,
            generated: "increment",
        },
        name: {
            type: "varchar",
        },
    },
    relations: {
        specialties: {
            type: "many-to-many",
            target: "Specialty",
            joinTable: {
                name: "VetSpecialty",
                joinColumn: {
                    name: "vet_id",
                    referencedColumnName: "id",
                },
                inverseJoinColumn: {
                    name: "specialty_name",
                    referencedColumnName: "name",
                },
            },
        },
    },
})

export default [
    specialtyEntitySchema,
    vetSpecialtyEntitySchema,
    vetEntitySchema,
]